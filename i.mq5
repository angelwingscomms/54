//+------------------------------------------------------------------+
//| TKAN_Trader.mq5                                                  |
//| Expert Advisor — ONNX price-forecast model → market orders       |
//+------------------------------------------------------------------+
#property copyright "TKAN Trader"
#property version   "1.07"
#property strict

#include <Trade\Trade.mqh>
#include "models/btc1h/model_meta.mqh" 

#resource "\\Experts\\54\\models\\btc1h\\model.onnx" as uchar ExtModel[]

//── Inputs ─────────────────────────────────────────────────────────
input group "=== Trade ==="
input double InpLots         = 0.1;    // Lot size
input double InpSL_ATR_Mult  = 1.5;    // SL = ATR(14) × this multiplier
input double InpTP_ATR_Mult  = 0.0;    // TP via ATR (0 = use model forecast price)
input double InpMinMoveRatio = 0.3;    // Min |forecast−price| / ATR to open trade
input int    InpMagic        = 202501; // Magic number

input group "=== Range Risk ==="
input int    InpMaxPredictedAdversePoints = 5400; // Max predicted adverse move
input int    InpMinSLPoints               = 0;    // 0 = no manual minimum
input int    InpMaxSLPoints               = 0;    // 0 = no manual maximum
input int    InpSLBufferPoints            = 0;    // Put SL beyond predicted range when > 0
input double InpStopoutSafetyFactor       = 0.75; // Require cushion above stopout
input bool   InpVerboseLogs               = true; // Verbose decision logs

//── Globals ─────────────────────────────────────────────────────────
static long   g_onnx         = INVALID_HANDLE;
static CTrade g_trade;
static datetime g_last_bar   = 0;
static int    g_atr_handle   = INVALID_HANDLE;

int g_rolling_bars;
int g_hist_needed;
int g_target_close_idx = -1;
int g_target_up_idx    = -1;
int g_target_down_idx  = -1;

bool NormaliseFeatureName(const string feature, string &out)
{
   if(feature == NULL)
      return false;

   out = feature;
   StringToLower(out);
   StringTrimLeft(out);
   StringTrimRight(out);

   if(out == "" || out == "none" || out == "null")
      out = "close";

   return (out == "close" || out == "open" || out == "high" || out == "low" ||
           out == "tick_volume" || out == "volume" || out == "vol" ||
           out == "real_volume" || out == "spread");
}

string CleanText(string value)
{
   StringToLower(value);
   StringTrimLeft(value);
   StringTrimRight(value);
   return value;
}

int FindTargetIndex(const string target)
{
   string wanted = CleanText(target);
   for(int i = 0; i < ArraySize(MODEL_TARGET_NAMES); i++)
   {
      string got = CleanText(MODEL_TARGET_NAMES[i]);
      if(got == wanted)
         return i;
   }
   return -1;
}

bool IsPrimaryFeature(const int index, const string feature)
{
   if(index < 0 || index >= MODEL_N_FEATURES)
      return false;

   string sym = MODEL_SYMBOLS[index];
   StringTrimLeft(sym);
   StringTrimRight(sym);

   string f;
   if(!NormaliseFeatureName(MODEL_FEATURE_NAMES[index], f))
      return false;

   return (sym == MODEL_PRIMARY_SYMBOL && f == feature);
}

bool ValidateModelMetadata()
{
   if(MODEL_SEQ_LEN <= 0 || MODEL_N_FEATURES <= 0 || MODEL_N_OUT <= 0 || MODEL_N_AHEAD <= 0)
   {
      Print("ERROR: invalid model dimensions in model_meta.mqh");
      return false;
   }

   if(ArraySize(MODEL_SYMBOLS) != MODEL_N_FEATURES || ArraySize(MODEL_FEATURE_NAMES) != MODEL_N_FEATURES)
   {
      PrintFormat("ERROR: metadata array size mismatch. symbols=%d features=%d expected=%d",
                  ArraySize(MODEL_SYMBOLS), ArraySize(MODEL_FEATURE_NAMES), MODEL_N_FEATURES);
      return false;
   }

   if(ArraySize(MODEL_X_MIN) != MODEL_N_FEATURES || ArraySize(MODEL_X_SCALE) != MODEL_N_FEATURES)
   {
      PrintFormat("ERROR: X scaler size mismatch. min=%d scale=%d expected=%d",
                  ArraySize(MODEL_X_MIN), ArraySize(MODEL_X_SCALE), MODEL_N_FEATURES);
      return false;
   }

   if(MODEL_TARGET_MODE != "range_close" || MODEL_N_AHEAD != 1 || MODEL_N_OUT != 3)
   {
      PrintFormat("ERROR: this EA requires range_close metadata with n_ahead=1 and n_out=3. mode=%s n_ahead=%d n_out=%d",
                  MODEL_TARGET_MODE, MODEL_N_AHEAD, MODEL_N_OUT);
      return false;
   }

   if(ArraySize(MODEL_TARGET_NAMES) != MODEL_N_OUT)
   {
      PrintFormat("ERROR: target metadata size mismatch. target_names=%d expected=%d",
                  ArraySize(MODEL_TARGET_NAMES), MODEL_N_OUT);
      return false;
   }

   if(ArraySize(MODEL_Y_MIN) != MODEL_N_OUT || ArraySize(MODEL_Y_SCALE) != MODEL_N_OUT)
   {
      PrintFormat("ERROR: Y scaler size mismatch. min=%d scale=%d expected=%d",
                  ArraySize(MODEL_Y_MIN), ArraySize(MODEL_Y_SCALE), MODEL_N_OUT);
      return false;
   }

   for(int i = 0; i < MODEL_N_FEATURES; i++)
   {
      string sym = MODEL_SYMBOLS[i];
      if(sym == NULL)
      {
         PrintFormat("ERROR: MODEL_SYMBOLS[%d] is NULL", i);
         return false;
      }

      StringTrimLeft(sym);
      StringTrimRight(sym);
      if(StringLen(sym) == 0)
      {
         PrintFormat("ERROR: MODEL_SYMBOLS[%d] is empty", i);
         return false;
      }

      string f;
      if(!NormaliseFeatureName(MODEL_FEATURE_NAMES[i], f))
      {
         PrintFormat("ERROR: unsupported feature metadata at index %d: symbol='%s' feature='%s'",
                     i, sym, MODEL_FEATURE_NAMES[i]);
         return false;
      }
   }

   return true;
}

//+------------------------------------------------------------------+
//| OnInit                                                           |
//+------------------------------------------------------------------+
int OnInit()
{
   g_rolling_bars = MODEL_ROLLING_DAYS * 24;
   g_hist_needed  = g_rolling_bars + MODEL_N_AHEAD + MODEL_SEQ_LEN;

   if(!ValidateModelMetadata())
      return INIT_FAILED;

   g_target_close_idx = FindTargetIndex("close");
   g_target_up_idx    = FindTargetIndex("up_range");
   g_target_down_idx  = FindTargetIndex("down_range");
   if(g_target_close_idx < 0 || g_target_up_idx < 0 || g_target_down_idx < 0)
   {
      Print("ERROR: required targets are missing. Need close, up_range, down_range.");
      return INIT_FAILED;
   }

   g_onnx = OnnxCreateFromBuffer(ExtModel, ONNX_DEFAULT);
   if(g_onnx == INVALID_HANDLE)
   {
      Print("ERROR: OnnxCreateFromBuffer failed: ", GetLastError());
      return INIT_FAILED;
   }

   ulong in_shape[]  = {1, (ulong)(MODEL_SEQ_LEN * MODEL_N_FEATURES)};
   ulong out_shape[] = {1, (ulong)MODEL_N_OUT}; 

   if(!OnnxSetInputShape(g_onnx, 0, in_shape))
   {
      Print("ERROR: ONNX Input Shape setup failed: ", GetLastError());
      return INIT_FAILED;
   }
   
   if(!OnnxSetOutputShape(g_onnx, 0, out_shape))
   {
      Print("ERROR: ONNX Output Shape setup failed: ", GetLastError());
      return INIT_FAILED;
   }

   g_atr_handle = iATR(_Symbol, PERIOD_H1, 14);
   if(g_atr_handle == INVALID_HANDLE)
   {
      Print("ERROR: iATR failed: ", GetLastError());
      return INIT_FAILED;
   }

   g_trade.SetExpertMagicNumber(InpMagic);
   g_trade.SetDeviationInPoints(20);
   g_trade.SetTypeFillingBySymbol(_Symbol);

   Print("TKAN_Trader initialised. Features=", MODEL_N_FEATURES,
         "  hist_needed=", g_hist_needed,
         "  rolling_bars=", g_rolling_bars,
         "  n_ahead=", MODEL_N_AHEAD,
         "  target_mode=", MODEL_TARGET_MODE,
         "  primary=", MODEL_PRIMARY_SYMBOL);
         
   return INIT_SUCCEEDED;
}

void OnDeinit(const int reason)
{
   if(g_onnx != INVALID_HANDLE)       { OnnxRelease(g_onnx);              g_onnx       = INVALID_HANDLE; }
   if(g_atr_handle != INVALID_HANDLE) { IndicatorRelease(g_atr_handle);   g_atr_handle = INVALID_HANDLE; }
}

//+------------------------------------------------------------------+
//| Robust History Fetcher                                           |
//+------------------------------------------------------------------+
bool CopyFeatureAtShift(const string sym, const string feature, int shift, double &value)
{
   int copied = 0;

   if(feature == "close")
   {
      double buf[];
      copied = CopyClose(sym, PERIOD_H1, shift, 1, buf);
      if(copied == 1) value = buf[0];
   }
   else if(feature == "open")
   {
      double buf[];
      copied = CopyOpen(sym, PERIOD_H1, shift, 1, buf);
      if(copied == 1) value = buf[0];
   }
   else if(feature == "high")
   {
      double buf[];
      copied = CopyHigh(sym, PERIOD_H1, shift, 1, buf);
      if(copied == 1) value = buf[0];
   }
   else if(feature == "low")
   {
      double buf[];
      copied = CopyLow(sym, PERIOD_H1, shift, 1, buf);
      if(copied == 1) value = buf[0];
   }
   else if(feature == "tick_volume" || feature == "volume" || feature == "vol")
   {
      long vol[];
      copied = CopyTickVolume(sym, PERIOD_H1, shift, 1, vol);
      if(copied == 1) value = (double)vol[0];
   }
   else if(feature == "real_volume")
   {
      long vol[];
      copied = CopyRealVolume(sym, PERIOD_H1, shift, 1, vol);
      if(copied == 1) value = (double)vol[0];
   }
   else if(feature == "spread")
   {
      int spread[];
      copied = CopySpread(sym, PERIOD_H1, shift, 1, spread);
      if(copied == 1) value = (double)spread[0];
   }

   if(copied != 1)
   {
      PrintFormat("History copy failed for %s_%s shift=%d copied=%d error=%d",
                  sym, feature, shift, copied, GetLastError());
      return false;
   }

   return true;
}

bool GetFeatureHistory(const string sym, const string feature, const datetime &base_times[], int length, double &out[])
{
   ArrayResize(out, length);

   string f;
   if(!NormaliseFeatureName(feature, f))
   {
      PrintFormat("Unsupported feature mapping detected: '%s'", feature);
      return false;
   }

   if(!SymbolSelect(sym, true))
   {
      PrintFormat("SymbolSelect failed for %s error=%d", sym, GetLastError());
      return false;
   }

   for(int i = 0; i < length; i++)
   {
      ResetLastError();
      int shift = iBarShift(sym, PERIOD_H1, base_times[i], false);
      if(shift < 0)
      {
         PrintFormat("No aligned history for %s_%s at %s error=%d",
                     sym, f, TimeToString(base_times[i]), GetLastError());
         return false;
      }

      if(!CopyFeatureAtShift(sym, f, shift, out[i]))
         return false;
   }

   return true;
}

bool RollingMedianAt(const double &raw[], int pos, double &median_out)
{
   int total = ArraySize(raw);
   int med_end = pos - MODEL_N_AHEAD;
   int med_start = med_end - g_rolling_bars + 1;

   if(med_start < 0 || med_end >= total)
      return false;

   double tmp[];
   ArrayResize(tmp, g_rolling_bars);
   for(int j = 0; j < g_rolling_bars; j++)
      tmp[j] = raw[med_start + j];
   ArraySort(tmp);

   median_out = (g_rolling_bars % 2 == 0) ?
                (tmp[g_rolling_bars / 2 - 1] + tmp[g_rolling_bars / 2]) * 0.5 :
                tmp[g_rolling_bars / 2];

   return true;
}

bool NormaliseWithRollingMedian(const double &raw[], double &norm[])
{
   int total = ArraySize(raw);
   if(total < g_hist_needed) return false;

   ArrayResize(norm, MODEL_SEQ_LEN);
   int offset = total - MODEL_SEQ_LEN;

   for(int i = 0; i < MODEL_SEQ_LEN; i++)
   {
      int pos = offset + i;
      double med;
      if(!RollingMedianAt(raw, pos, med))
         return false;

      norm[i] = (med > 0.0) ? raw[pos] / med : 0.0;
   }
   return true;
}

bool RunModel(const double &input_matrix[], double &predictions[])
{
   int cells = MODEL_SEQ_LEN * MODEL_N_FEATURES;
   float in_buf[];
   ArrayResize(in_buf, cells);
   for(int i = 0; i < cells; i++) in_buf[i] = (float)input_matrix[i];

   float out_buf[];
   ArrayResize(out_buf, MODEL_N_OUT);

   if(!OnnxRun(g_onnx, ONNX_NO_CONVERSION, in_buf, out_buf))
   {
      Print("OnnxRun error: ", GetLastError());
      return false;
   }

   ArrayResize(predictions, MODEL_N_OUT);
   for(int i = 0; i < MODEL_N_OUT; i++)
      predictions[i] = (double)out_buf[i];

   return true;
}

bool HasOpenPosition()
{
   for(int i = PositionsTotal() - 1; i >= 0; i--)
      if(PositionGetSymbol(i) == _Symbol && PositionGetInteger(POSITION_MAGIC) == InpMagic)
         return true;
   return false;
}

double NormaliseVolume(double lots)
{
   double min_lot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double max_lot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
   double step    = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);

   if(step <= 0.0)
      return lots;

   lots = MathMax(min_lot, MathMin(max_lot, lots));
   lots = min_lot + MathRound((lots - min_lot) / step) * step;
   lots = MathMax(min_lot, MathMin(max_lot, lots));

   int digits = 0;
   double probe = step;
   while(probe < 1.0 && digits < 8)
   {
      probe *= 10.0;
      digits++;
   }

   return NormalizeDouble(lots, digits);
}

double NormalisePrice(double price)
{
   double tick = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
   if(tick <= 0.0)
      tick = _Point;
   return MathRound(price / tick) * tick;
}

double PointsToPrice(const double points)
{
   return points * _Point;
}

double PriceDistancePoints(const double price_a, const double price_b)
{
   if(_Point <= 0.0)
      return 0.0;
   return MathAbs(price_a - price_b) / _Point;
}

bool CheckStopoutSafety(const bool is_buy,
                        const double lots,
                        const double entry,
                        const double sl,
                        double &order_margin,
                        double &sl_profit,
                        double &equity_after,
                        double &margin_after,
                        double &free_after,
                        double &level_after,
                        string &reason)
{
   reason = "";
   order_margin = 0.0;
   sl_profit = 0.0;
   equity_after = 0.0;
   margin_after = 0.0;
   free_after = 0.0;
   level_after = 0.0;

   ENUM_ORDER_TYPE order_type = is_buy ? ORDER_TYPE_BUY : ORDER_TYPE_SELL;

   if(!OrderCalcProfit(order_type, _Symbol, lots, entry, sl, sl_profit))
   {
      reason = StringFormat("OrderCalcProfit failed error=%d", GetLastError());
      return false;
   }

   if(!OrderCalcMargin(order_type, _Symbol, lots, entry, order_margin))
   {
      reason = StringFormat("OrderCalcMargin failed error=%d", GetLastError());
      return false;
   }

   double equity = AccountInfoDouble(ACCOUNT_EQUITY);
   double free_margin = AccountInfoDouble(ACCOUNT_MARGIN_FREE);
   double used_margin = AccountInfoDouble(ACCOUNT_MARGIN);

   equity_after = equity + sl_profit;
   margin_after = used_margin + order_margin;
   free_after = free_margin - order_margin + sl_profit;

   if(margin_after > 0.0)
      level_after = equity_after / margin_after * 100.0;
   else
      level_after = 1.0e100;

   if(equity_after <= 0.0)
   {
      reason = StringFormat("equity after SL would be %.2f", equity_after);
      return false;
   }

   if(free_after <= 0.0)
   {
      reason = StringFormat("free margin after SL would be %.2f", free_after);
      return false;
   }

   double stopout = AccountInfoDouble(ACCOUNT_MARGIN_SO_SO);
   if(stopout <= 0.0)
      return true;

   double safety = InpStopoutSafetyFactor;
   if(safety <= 0.0 || safety > 1.0)
      safety = 1.0;

   long stopout_mode = AccountInfoInteger(ACCOUNT_MARGIN_SO_MODE);
   if(stopout_mode == ACCOUNT_STOPOUT_MODE_PERCENT)
   {
      double required_level = stopout / safety;
      if(level_after <= required_level)
      {
         reason = StringFormat("margin level after SL %.2f%% <= required %.2f%%", level_after, required_level);
         return false;
      }
   }
   else
   {
      double required_free = stopout / safety;
      if(free_after <= required_free)
      {
         reason = StringFormat("free margin after SL %.2f <= required %.2f", free_after, required_free);
         return false;
      }
   }

   return true;
}

void OnTick()
{
   datetime bar_time = iTime(_Symbol, PERIOD_H1, 0);
   if(bar_time == g_last_bar) return;
   bool first_run = (g_last_bar == 0);
   g_last_bar = bar_time;

   double atr_buf[];
   if(CopyBuffer(g_atr_handle, 0, 1, 1, atr_buf) < 1) return;
   double atr = atr_buf[0];
   if(atr <= 0.0) return;

   datetime base_times[];
   ArrayResize(base_times, g_hist_needed);
   int copied_times = CopyTime(_Symbol, PERIOD_H1, 1, g_hist_needed, base_times);
   if(copied_times < g_hist_needed)
   {
      PrintFormat("Not enough base history times for %s (got %d, need %d)",
                  _Symbol, copied_times, g_hist_needed);
      return;
   }

   //── 1. Loop through ALL symbols * features matching dimensionality ────────
   double raw_all[];
   ArrayResize(raw_all, g_hist_needed * MODEL_N_FEATURES);

   for(int s = 0; s < MODEL_N_FEATURES; s++)
   {
      double rawS[];
      if(!GetFeatureHistory(MODEL_SYMBOLS[s], MODEL_FEATURE_NAMES[s], base_times, g_hist_needed, rawS)) return;
      for(int t = 0; t < g_hist_needed; t++)
         raw_all[t * MODEL_N_FEATURES + s] = rawS[t];
   }

   //── 2. Rolling-median normalise each feature ──────────────────
   double median_target = 1.0;
   bool have_target_median = false;
   double seq_matrix[];
   ArrayResize(seq_matrix, MODEL_SEQ_LEN * MODEL_N_FEATURES);

   for(int f = 0; f < MODEL_N_FEATURES; f++)
   {
      double col_raw[];
      ArrayResize(col_raw, g_hist_needed);
      for(int t = 0; t < g_hist_needed; t++)
         col_raw[t] = raw_all[t * MODEL_N_FEATURES + f];

      double col_norm[];
      if(!NormaliseWithRollingMedian(col_raw, col_norm)) return;

      if(IsPrimaryFeature(f, "close"))
      {
         if(!RollingMedianAt(col_raw, g_hist_needed, median_target))
            return;
         have_target_median = true;
      }

      for(int t = 0; t < MODEL_SEQ_LEN; t++)
         seq_matrix[t * MODEL_N_FEATURES + f] = col_norm[t];
   }

   if(!have_target_median || median_target <= 0.0)
   {
      Print("No primary close median available for model output inversion.");
      return;
   }

   //── 3. Apply MinMax scaling to X ───
   for(int t = 0; t < MODEL_SEQ_LEN; t++)
      for(int f = 0; f < MODEL_N_FEATURES; f++)
      {
         int idx = t * MODEL_N_FEATURES + f;
         double xs = MODEL_X_SCALE[f];
         seq_matrix[idx] = (xs != 0.0) ? (seq_matrix[idx] - MODEL_X_MIN[f]) / xs : 0.0;
      }

   //── 4. ONNX inference ─────────────────────────────────────────
   double y_scaled[];
   if(!RunModel(seq_matrix, y_scaled)) return;

   //── 5. Inverse MinMax on Y using generated .mqh arrays ────────
   double y_norm[];
   ArrayResize(y_norm, MODEL_N_OUT);
   for(int i = 0; i < MODEL_N_OUT; i++)
      y_norm[i] = y_scaled[i] * MODEL_Y_SCALE[i] + MODEL_Y_MIN[i];

   //── 6. Reconstruct current-bar close/high/low forecasts ───────
   double forecast_price = y_norm[g_target_close_idx] * median_target;
   double pred_up_range  = MathMax(y_norm[g_target_up_idx] * median_target, 0.0);
   double pred_down_range = MathMax(y_norm[g_target_down_idx] * median_target, 0.0);

   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double mid = (ask + bid) * 0.5;
   double spread_points = (ask - bid) / _Point;

   double bar_open = iOpen(_Symbol, PERIOD_H1, 0);
   double bar_high = iHigh(_Symbol, PERIOD_H1, 0);
   double bar_low  = iLow(_Symbol, PERIOD_H1, 0);
   if(bar_open <= 0.0 || bar_high <= 0.0 || bar_low <= 0.0 || ask <= 0.0 || bid <= 0.0)
   {
      PrintFormat("Invalid current-bar market data. open=%.5f high=%.5f low=%.5f bid=%.5f ask=%.5f",
                  bar_open, bar_high, bar_low, bid, ask);
      return;
   }

   double raw_pred_high = bar_open + pred_up_range;
   double raw_pred_low  = bar_open - pred_down_range;
   double pred_high = MathMax(raw_pred_high, MathMax(bar_open, forecast_price));
   double pred_low  = MathMin(raw_pred_low, MathMin(bar_open, forecast_price));
   bool range_adjusted = (pred_high != raw_pred_high || pred_low != raw_pred_low);

   double move      = forecast_price - mid;
   double min_move  = InpMinMoveRatio * atr;

   if(InpVerboseLogs)
   {
      PrintFormat("Forecast trigger=%s bar=%s bid=%.5f ask=%.5f mid=%.5f spread_pts=%.1f ATR=%.5f open=%.5f high=%.5f low=%.5f",
                  first_run ? "attach_or_first_tick" : "new_bar",
                  TimeToString(bar_time), bid, ask, mid, spread_points, atr, bar_open, bar_high, bar_low);
      PrintFormat("Predictions close=%.5f high=%.5f low=%.5f up_range=%.5f down_range=%.5f median=%.5f adjusted_range=%s",
                  forecast_price, pred_high, pred_low, pred_up_range, pred_down_range, median_target,
                  range_adjusted ? "true" : "false");
   }

   if(HasOpenPosition())
   {
      if(InpVerboseLogs)
         Print("No trade: existing position with this symbol and magic number.");
      return;
   }

   if(MathAbs(move) < min_move)
   {
      if(InpVerboseLogs)
         PrintFormat("No trade: forecast move too small. move=%.5f min_move=%.5f ratio=%.3f",
                     move, min_move, InpMinMoveRatio);
      return;
   }

   bool is_buy = (move > 0.0);
   if(is_buy)
   {
      if(bar_high >= forecast_price || mid >= forecast_price)
      {
         if(InpVerboseLogs)
            PrintFormat("No trade: buy target already touched this bar. high=%.5f mid=%.5f target=%.5f; waiting for next bar.",
                        bar_high, mid, forecast_price);
         return;
      }
   }
   else
   {
      if(bar_low <= forecast_price || mid <= forecast_price)
      {
         if(InpVerboseLogs)
            PrintFormat("No trade: sell target already touched this bar. low=%.5f mid=%.5f target=%.5f; waiting for next bar.",
                        bar_low, mid, forecast_price);
         return;
      }
   }

   double adverse_points = 0.0;
   if(is_buy && pred_low < mid)
      adverse_points = (mid - pred_low) / _Point;
   else if(!is_buy && pred_high > mid)
      adverse_points = (pred_high - mid) / _Point;

   if(InpMaxPredictedAdversePoints > 0 && adverse_points > (double)InpMaxPredictedAdversePoints)
   {
      if(InpVerboseLogs)
         PrintFormat("No trade: predicted adverse move too large. side=%s adverse_pts=%.1f limit_pts=%d pred_low=%.5f pred_high=%.5f mid=%.5f",
                     is_buy ? "buy" : "sell", adverse_points, InpMaxPredictedAdversePoints, pred_low, pred_high, mid);
      return;
   }

   if(InpVerboseLogs)
      PrintFormat("Trade candidate accepted by target/adverse filters. side=%s move=%.5f min_move=%.5f adverse_pts=%.1f limit_pts=%d",
                  is_buy ? "buy" : "sell", move, min_move, adverse_points, InpMaxPredictedAdversePoints);

   //── 8. SL / TP ────────────────────────────────────────────────
   double tick = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
   if(tick <= 0.0) tick = _Point;
   double min_stop = (double)SymbolInfoInteger(_Symbol, SYMBOL_TRADE_STOPS_LEVEL) * _Point;
   if(min_stop < tick) min_stop = tick;

   double min_sl_dist = MathMax(min_stop, InpMinSLPoints > 0 ? PointsToPrice((double)InpMinSLPoints) : 0.0);
   double max_sl_dist = InpMaxSLPoints > 0 ? PointsToPrice((double)InpMaxSLPoints) : 0.0;
   double sl_dist = MathMax(InpSL_ATR_Mult * atr, min_sl_dist);

   if(max_sl_dist > 0.0 && sl_dist > max_sl_dist)
   {
      if(InpVerboseLogs)
         PrintFormat("No trade: required SL distance exceeds max. required_pts=%.1f max_pts=%d",
                     sl_dist / _Point, InpMaxSLPoints);
      return;
   }

   if(InpSLBufferPoints > 0)
   {
      double buffer = PointsToPrice((double)InpSLBufferPoints);
      double range_sl = is_buy ? pred_low - buffer : pred_high + buffer;
      double range_sl_dist = is_buy ? ask - range_sl : range_sl - bid;
      if(range_sl_dist > sl_dist)
      {
         if(max_sl_dist > 0.0 && range_sl_dist > max_sl_dist)
         {
            if(InpVerboseLogs)
               PrintFormat("No trade: predicted-range SL exceeds max. range_sl_pts=%.1f max_pts=%d",
                           range_sl_dist / _Point, InpMaxSLPoints);
            return;
         }
         sl_dist = range_sl_dist;
      }
   }

   double entry = is_buy ? ask : bid;
   double sl = is_buy ? entry - sl_dist : entry + sl_dist;
   double tp = 0.0;
   if(InpTP_ATR_Mult > 0.0)
      tp = is_buy ? ask + InpTP_ATR_Mult * atr : bid - InpTP_ATR_Mult * atr;
   else
   {
      tp = forecast_price;
      if(is_buy && tp <= ask + min_stop)
      {
         if(InpVerboseLogs)
            PrintFormat("No trade: forecast TP is too close for buy. tp=%.5f ask=%.5f min_stop_pts=%.1f",
                        tp, ask, min_stop / _Point);
         return;
      }
      if(!is_buy && tp >= bid - min_stop)
      {
         if(InpVerboseLogs)
            PrintFormat("No trade: forecast TP is too close for sell. tp=%.5f bid=%.5f min_stop_pts=%.1f",
                        tp, bid, min_stop / _Point);
         return;
      }
   }

   if(is_buy)
   {
      sl = MathMin(sl, bid - min_stop);
      tp = MathMax(tp, ask + min_stop);
   }
   else
   {
      sl = MathMax(sl, ask + min_stop);
      tp = MathMin(tp, bid - min_stop);
   }

   sl = NormalisePrice(sl);
   tp = NormalisePrice(tp);

   double actual_sl_points = PriceDistancePoints(entry, sl);
   if(max_sl_dist > 0.0 && actual_sl_points > (double)InpMaxSLPoints + 0.5)
   {
      if(InpVerboseLogs)
         PrintFormat("No trade: broker-adjusted SL exceeds max. sl_pts=%.1f max_pts=%d sl=%.5f entry=%.5f",
                     actual_sl_points, InpMaxSLPoints, sl, entry);
      return;
   }

   //── 9. Place order ───────────────────────────────────────────
   double lots = NormaliseVolume(InpLots);
   double order_margin, sl_profit, equity_after, margin_after, free_after, level_after;
   string stopout_reason;
   bool stopout_ok = CheckStopoutSafety(is_buy, lots, entry, sl,
                                        order_margin, sl_profit, equity_after,
                                        margin_after, free_after, level_after,
                                        stopout_reason);

   if(InpVerboseLogs)
      PrintFormat("Stopout check ok=%s lots=%.4f sl_pts=%.1f margin=%.2f sl_profit=%.2f equity_after=%.2f margin_after=%.2f free_after=%.2f level_after=%.2f reason=%s",
                  stopout_ok ? "true" : "false", lots, actual_sl_points, order_margin, sl_profit,
                  equity_after, margin_after, free_after, level_after, stopout_reason);

   if(!stopout_ok)
   {
      PrintFormat("No trade: stopout safety check failed. %s", stopout_reason);
      return;
   }

   string comment = StringFormat("TKAN fc=%.2f adv=%.0f", forecast_price, adverse_points);
   bool sent = is_buy ? g_trade.Buy(lots,  _Symbol, ask, sl, tp, comment) :
                        g_trade.Sell(lots, _Symbol, bid, sl, tp, comment);

   uint ret = g_trade.ResultRetcode();
   if(!sent || (ret != TRADE_RETCODE_DONE && ret != TRADE_RETCODE_DONE_PARTIAL && ret != TRADE_RETCODE_PLACED))
      PrintFormat("Trade failed. sent=%s retcode=%u %s lots=%.4f sl=%.5f tp=%.5f",
                  sent ? "true" : "false", ret, g_trade.ResultRetcodeDescription(), lots, sl, tp);
   else
      PrintFormat("Trade sent. retcode=%u %s lots=%.4f sl=%.5f tp=%.5f",
                  ret, g_trade.ResultRetcodeDescription(), lots, sl, tp);
}
