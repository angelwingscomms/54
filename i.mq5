//+------------------------------------------------------------------+
//| TKAN_Trader.mq5                                                  |
//| Expert Advisor — ONNX price-forecast model → market orders       |
//+------------------------------------------------------------------+
#property copyright "TKAN Trader"
#property version   "1.05"
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

//+------------------------------------------------------------------+
//| OnInit                                                           |
//+------------------------------------------------------------------+
int OnInit()
{
   g_rolling_bars = MODEL_ROLLING_DAYS * 24;  
   g_hist_needed  = g_rolling_bars + MODEL_N_AHEAD + MODEL_SEQ_LEN;

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
bool GetFeatureHistory(const string sym, const string feature, int length, double &out[])
{
   ArrayResize(out, length);
   int copied = 0;
   
   // Clean format and lowercase
   string f = feature;
   StringToLower(f);
   StringTrimLeft(f);
   StringTrimRight(f);
   
   // Robust Fallback Map
   if(f == "close" || f == "" || f == "none") 
      copied = CopyClose(sym, PERIOD_H1, 1, length, out);
   else if(f == "open") 
      copied = CopyOpen(sym, PERIOD_H1, 1, length, out);
   else if(f == "high") 
      copied = CopyHigh(sym, PERIOD_H1, 1, length, out);
   else if(f == "low") 
      copied = CopyLow(sym, PERIOD_H1, 1, length, out);
   else if(f == "tick_volume" || f == "volume" || f == "vol") {
      long vol[];
      copied = CopyTickVolume(sym, PERIOD_H1, 1, length, vol);
      for(int i = 0; i < copied; i++) out[i] = (double)vol[i];
   }
   else if(f == "real_volume") {
      long vol[];
      copied = CopyRealVolume(sym, PERIOD_H1, 1, length, vol);
      for(int i = 0; i < copied; i++) out[i] = (double)vol[i];
   }
   else {
      PrintFormat("Unsupported feature mapping detected: '%s'", feature);
      return false;
   }
   
   if(copied < length)
   {
      PrintFormat("Not enough history for %s_%s (got %d, need %d)", sym, f, copied, length);
      return false;
   }
   ArrayReverse(out);
   return true;
}

bool NormaliseWithRollingMedian(const double &raw[], double &norm[], double &median_out)
{
   int total = ArraySize(raw);
   if(total < g_hist_needed) return false;

   ArrayResize(norm, MODEL_SEQ_LEN);
   int offset = total - MODEL_SEQ_LEN;

   for(int i = 0; i < MODEL_SEQ_LEN; i++)
   {
      int pos       = offset + i;
      int med_end   = pos - MODEL_N_AHEAD; 
      int med_start = med_end - g_rolling_bars;

      if(med_start < 0) return false;

      double tmp[];
      ArrayResize(tmp, g_rolling_bars);
      for(int j = 0; j < g_rolling_bars; j++) tmp[j] = raw[med_start + j];
      ArraySort(tmp);

      double med = (g_rolling_bars % 2 == 0) ? 
                   (tmp[g_rolling_bars/2 - 1] + tmp[g_rolling_bars/2]) * 0.5 : 
                   tmp[g_rolling_bars / 2];

      norm[i] = (med > 0.0) ? raw[pos] / med : 0.0;
      if(i == MODEL_SEQ_LEN - 1) median_out = med;
   }
   return true;
}

double RunModel(const double &input_matrix[])
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
      return -1.0;
   }
   return (double)out_buf[0];
}

bool HasOpenPosition()
{
   for(int i = PositionsTotal() - 1; i >= 0; i--)
      if(PositionGetSymbol(i) == _Symbol && PositionGetInteger(POSITION_MAGIC) == InpMagic)
         return true;
   return false;
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

   //── 1. Loop through ALL symbols * features matching dimensionality ────────
   double raw_all[];
   ArrayResize(raw_all, g_hist_needed * MODEL_N_FEATURES);

   for(int s = 0; s < MODEL_N_FEATURES; s++)
   {
      double rawS[];
      if(!GetFeatureHistory(MODEL_SYMBOLS[s], MODEL_FEATURE_NAMES[s], g_hist_needed, rawS)) return;
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

      double col_norm[], col_median;
      if(!NormaliseWithRollingMedian(col_raw, col_norm, col_median)) return;

      if(f == 0) median_target = col_median; // Target is always index 0

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
   double y_scaled = RunModel(seq_matrix);
   if(y_scaled < 0.0) return;

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
   sl = MathRound(sl / tick) * tick;
   tp = MathRound(tp / tick) * tick;

   // Ensure TP is valid (prevent ERR_TRADE_INVALID_STOPS if prediction is inside spread)
   if(move > 0.0 && tp <= ask) tp = ask + tick * 10;
   if(move < 0.0 && tp >= bid) tp = bid - tick * 10;

   //── 9. Place order ───────────────────────────────────────────
   string comment = StringFormat("TKAN fc=%.2f", forecast_price);
   if(move > 0.0) g_trade.Buy(InpLots,  _Symbol, ask, sl, tp, comment);
   else           g_trade.Sell(InpLots, _Symbol, bid, sl, tp, comment);
}