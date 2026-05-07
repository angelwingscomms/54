//+------------------------------------------------------------------+
//| TKAN_Trader.mq5                                                  |
//| Expert Advisor — ONNX price-forecast model → market orders       |
//+------------------------------------------------------------------+
#property copyright "TKAN Trader"
#property version   "1.06"
#property strict

#include <Trade\Trade.mqh>
#include "model_meta.mqh" 

#resource "model.onnx" as uchar ExtModel[]

//── Inputs ─────────────────────────────────────────────────────────
input group "=== Trade ==="
input double InpLots         = 0.1;    // Lot size
input double InpSL_ATR_Mult  = 1.5;    // SL = ATR(14) × this multiplier
input double InpTP_ATR_Mult  = 0.0;    // TP via ATR (0 = use model forecast price)
input double InpMinMoveRatio = 0.3;    // Min |forecast−price| / ATR to open trade
input int    InpMagic        = 202501; // Magic number

//── Globals ─────────────────────────────────────────────────────────
static long   g_onnx         = INVALID_HANDLE;
static CTrade g_trade;
static datetime g_last_bar   = 0;
static int    g_atr_handle   = INVALID_HANDLE;

int g_rolling_bars;
int g_hist_needed;

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

   if(ArraySize(MODEL_Y_MIN) < 1 || ArraySize(MODEL_Y_SCALE) < 1)
   {
      Print("ERROR: Y scaler metadata is empty");
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
         "  n_ahead=", MODEL_N_AHEAD);
         
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

bool RunModel(const double &input_matrix[], double &prediction)
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

   prediction = (double)out_buf[0];
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

void OnTick()
{
   datetime bar_time = iTime(_Symbol, PERIOD_H1, 0);
   if(bar_time == g_last_bar) return;
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

      if(f == 0 && !RollingMedianAt(col_raw, g_hist_needed, median_target))
         return;

      for(int t = 0; t < MODEL_SEQ_LEN; t++)
         seq_matrix[t * MODEL_N_FEATURES + f] = col_norm[t];
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
   double y_scaled;
   if(!RunModel(seq_matrix, y_scaled)) return;

   //── 5. Inverse MinMax on Y using generated .mqh arrays ────────
   double y_norm = y_scaled * MODEL_Y_SCALE[0] + MODEL_Y_MIN[0];

   //── 6. Multiply by rolling median → real-price forecast ───────
   double forecast_price = y_norm * median_target;

   //── 7. Signal direction ───────────────────────────────────────
   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double mid = (ask + bid) * 0.5;

   double move      = forecast_price - mid;
   double min_move  = InpMinMoveRatio * atr;

   PrintFormat("Bar=%s mid=%.5f fc=%.5f move=%.5f ATR=%.5f", TimeToString(bar_time), mid, forecast_price, move, atr);

   if(HasOpenPosition()) return;
   if(MathAbs(move) < min_move) return;

   //── 8. SL / TP ────────────────────────────────────────────────
   double sl_dist = InpSL_ATR_Mult * atr;
   double sl = (move > 0.0) ? ask - sl_dist : bid + sl_dist;
   double tp = (move > 0.0) ? ((InpTP_ATR_Mult > 0.0) ? ask + InpTP_ATR_Mult * atr : forecast_price) : 
                              ((InpTP_ATR_Mult > 0.0) ? bid - InpTP_ATR_Mult * atr : forecast_price);

   double tick = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
   if(tick <= 0.0) tick = _Point;
   double min_stop = (double)SymbolInfoInteger(_Symbol, SYMBOL_TRADE_STOPS_LEVEL) * _Point;
   if(min_stop < tick) min_stop = tick;

   if(move > 0.0)
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

   //── 9. Place order ───────────────────────────────────────────
   string comment = StringFormat("TKAN fc=%.2f", forecast_price);
   double lots = NormaliseVolume(InpLots);
   bool sent = (move > 0.0) ? g_trade.Buy(lots,  _Symbol, ask, sl, tp, comment) :
                              g_trade.Sell(lots, _Symbol, bid, sl, tp, comment);

   uint ret = g_trade.ResultRetcode();
   if(!sent || (ret != TRADE_RETCODE_DONE && ret != TRADE_RETCODE_DONE_PARTIAL && ret != TRADE_RETCODE_PLACED))
      PrintFormat("Trade failed. sent=%s retcode=%u %s lots=%.4f sl=%.5f tp=%.5f",
                  sent ? "true" : "false", ret, g_trade.ResultRetcodeDescription(), lots, sl, tp);
   else
      PrintFormat("Trade sent. retcode=%u %s lots=%.4f sl=%.5f tp=%.5f",
                  ret, g_trade.ResultRetcodeDescription(), lots, sl, tp);
}
