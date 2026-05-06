//+------------------------------------------------------------------+
//| TKAN_Trader.mq5                                                  |
//| Expert Advisor — ONNX price-forecast model → market orders       |
//|                                                                  |
//| Setup:                                                           |
//|   1. Copy model.onnx  → <MT5 data dir>\MQL5\Files\              |
//|   2. Copy model_meta.json (not read at runtime; values are hard- |
//|      coded as inputs — edit them to match your trained model).   |
//|   3. Attach EA to the target symbol on the H1 chart.            |
//|   4. Adjust inputs as needed and enable Algo Trading.            |
//+------------------------------------------------------------------+
#property copyright "TKAN Trader"
#property version   "1.00"
#property strict

#include <Trade\Trade.mqh>

//── Inputs ─────────────────────────────────────────────────────────
input group "=== Trade ==="
input double InpLots        = 0.1;   // Lot size
input double InpSL_ATR_Mult = 1.5;   // SL = ATR(14) × this multiplier
input double InpTP_ATR_Mult = 0.0;   // TP via ATR (0 = use model forecast price)
input double InpMinMoveRatio= 0.3;   // Min |forecast−price| / ATR to open trade
input int    InpMagic       = 202501; // Magic number

input group "=== Model (must match model_meta.json) ==="
input string InpOnnxFile    = "model.onnx"; // ONNX file in MQL5\Files\
input int    InpSeqLen      = 54;    // sequence_length
input int    InpNAhead      = 1;     // n_ahead
input int    InpRollingDays = 14;    // rolling_days
// Feature columns: primary symbol close, then additional symbols
// Edit these to match all_columns in model_meta.json (same order!)
input string InpSymCol0     = "";        // Extra symbol 1 (e.g. "$USDX"), blank=none
input string InpSymCol1     = "";        // Extra symbol 2 (e.g. "USDJPY"), blank=none

input group "=== X scaler (from model_meta.json → X_scaler) ==="
// Paste min_ and scale_ arrays here as comma-separated strings.
// If you trained with only XAUUSD close + 2 extra symbols, you have 3 features.
// Example with 3 features: "0.9980,0.9712,145.3"
input string InpXMin        = ""; // X_scaler min_ (one per feature, comma-sep)
input string InpXScale      = ""; // X_scaler scale_ (one per feature, comma-sep)

input group "=== Y scaler (from model_meta.json → y_scaler) ==="
input double InpYMin        = 0.0;   // y_scaler min_[0]
input double InpYScale      = 1.0;   // y_scaler scale_[0]

//── Globals ─────────────────────────────────────────────────────────
static long  g_onnx   = INVALID_HANDLE;
static CTrade g_trade;
static datetime g_last_bar = 0;
static int   g_atr_handle  = INVALID_HANDLE;

// Parsed scaler arrays
double g_x_min[];
double g_x_scale[];

// Rolling window length = rolling_bars + seq_len
int    g_rolling_bars;
int    g_hist_needed;
int    g_n_features;

//+------------------------------------------------------------------+
//| Parse a comma-separated string into a double array               |
//+------------------------------------------------------------------+
bool ParseDoubleArray(const string s, double &arr[])
{
   string parts[];
   int n = StringSplit(s, ',', parts);
   if(n <= 0) return false;
   ArrayResize(arr, n);
   for(int i = 0; i < n; i++)
      arr[i] = StringToDouble(parts[i]);
   return true;
}

//+------------------------------------------------------------------+
//| Count non-blank extra symbols                                    |
//+------------------------------------------------------------------+
int ExtraSymbolCount()
{
   int c = 0;
   if(InpSymCol0 != "") c++;
   if(InpSymCol1 != "") c++;
   return c;
}

string ExtraSymbol(int idx)
{
   if(idx == 0) return InpSymCol0;
   return InpSymCol1;
}

//+------------------------------------------------------------------+
//| OnInit                                                           |
//+------------------------------------------------------------------+
int OnInit()
{
   g_rolling_bars = InpRollingDays * 24;         // hourly bars
   g_hist_needed  = g_rolling_bars + InpSeqLen;
   g_n_features   = 1 + ExtraSymbolCount();       // primary close + extras

   //── Validate / parse scalers ──────────────────────────────────
   if(InpXMin == "" || InpXScale == "")
   {
      Print("ERROR: X scaler inputs are empty. "
            "Copy min_ and scale_ from model_meta.json.");
      return INIT_FAILED;
   }
   if(!ParseDoubleArray(InpXMin,   g_x_min)   ||
      !ParseDoubleArray(InpXScale, g_x_scale) ||
      ArraySize(g_x_min)   != g_n_features   ||
      ArraySize(g_x_scale) != g_n_features)
   {
      Print("ERROR: X scaler array length != n_features (", g_n_features, ")");
      return INIT_FAILED;
   }

   //── Load ONNX model ──────────────────────────────────────────
   g_onnx = OnnxCreate(InpOnnxFile, 0);
   if(g_onnx == INVALID_HANDLE)
   {
      Print("ERROR: OnnxCreate failed: ", GetLastError(),
            " — ensure ", InpOnnxFile, " is in MQL5\\Files\\");
      return INIT_FAILED;
   }

   // Input shape: [1, seq_len, n_features]
   ulong in_shape[]  = {1, (ulong)InpSeqLen, (ulong)g_n_features};
   ulong out_shape[] = {1, (ulong)InpNAhead};
   if(!OnnxSetInputShape(g_onnx, 0, in_shape))
   {
      Print("ERROR: OnnxSetInputShape failed: ", GetLastError());
      return INIT_FAILED;
   }
   if(!OnnxSetOutputShape(g_onnx, 0, out_shape))
   {
      Print("ERROR: OnnxSetOutputShape failed: ", GetLastError());
      return INIT_FAILED;
   }

   //── ATR indicator ────────────────────────────────────────────
   g_atr_handle = iATR(_Symbol, PERIOD_H1, 14);
   if(g_atr_handle == INVALID_HANDLE)
   {
      Print("ERROR: iATR failed: ", GetLastError());
      return INIT_FAILED;
   }

   g_trade.SetExpertMagicNumber(InpMagic);
   g_trade.SetDeviationInPoints(20);

   Print("TKAN_Trader initialised. Features=", g_n_features,
         "  hist_needed=", g_hist_needed);
   return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| OnDeinit                                                         |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   if(g_onnx != INVALID_HANDLE) { OnnxRelease(g_onnx); g_onnx = INVALID_HANDLE; }
   if(g_atr_handle != INVALID_HANDLE) { IndicatorRelease(g_atr_handle); }
}

//+------------------------------------------------------------------+
//| Collect close prices for one symbol, history length bars         |
//| Returns false if not enough bars                                 |
//+------------------------------------------------------------------+
bool GetCloseHistory(const string sym, int length, double &out[])
{
   ArrayResize(out, length);
   // index 0 = most recent closed bar
   int copied = CopyClose(sym, PERIOD_H1, 1, length, out);
   if(copied < length)
   {
      Print("Not enough history for ", sym, " (", copied, "/", length, ")");
      return false;
   }
   // CopyClose returns newest-first; reverse to chronological order
   ArrayReverse(out);
   return true;
}

//+------------------------------------------------------------------+
//| Rolling-median normalisation (matches Python pipeline)           |
//| raw[]   = chronological price series, length = hist_needed       |
//| norm[]  = last InpSeqLen normalised values (output)              |
//| median_last = median of the last rolling_bars raw prices (×curr) |
//+------------------------------------------------------------------+
bool NormaliseWithRollingMedian(const double &raw[], double &norm[], double &median_last)
{
   int total = ArraySize(raw); // should be hist_needed
   if(total < g_hist_needed) return false;

   // We need the rolling median at each of the last InpSeqLen positions.
   // Python: scaler_df = df.shift(n_ahead).rolling(rolling_bars).median()
   // For the EA (n_ahead = InpNAhead, we want the *input* sequence):
   // norm[t] = raw[t] / median(raw[t - rolling_bars .. t-1])  (shift-by-1)
   // The last InpSeqLen entries of raw[] map to the sequence window.

   ArrayResize(norm, InpSeqLen);
   int offset = total - InpSeqLen; // start of sequence window in raw[]

   for(int i = 0; i < InpSeqLen; i++)
   {
      int pos = offset + i;            // position in raw[] for this bar
      // median of raw[pos - rolling_bars .. pos - 1]  (exclude current bar)
      int med_start = pos - g_rolling_bars;
      if(med_start < 0) { Print("Not enough data for median at pos ", pos); return false; }

      double tmp[];
      ArrayResize(tmp, g_rolling_bars);
      for(int j = 0; j < g_rolling_bars; j++)
         tmp[j] = raw[med_start + j];
      ArraySort(tmp);
      double med;
      if(g_rolling_bars % 2 == 0)
         med = (tmp[g_rolling_bars/2 - 1] + tmp[g_rolling_bars/2]) * 0.5;
      else
         med = tmp[g_rolling_bars / 2];

      norm[i] = (med > 0.0) ? raw[pos] / med : 0.0;

      // Save the last bar's median for the target symbol (col 0)
      if(i == InpSeqLen - 1)
         median_last = med;
   }
   return true;
}

//+------------------------------------------------------------------+
//| Run ONNX inference                                               |
//| input_matrix: row-major [seq_len × n_features]  (already MinMax)|
//| Returns forecast price (real units) or -1 on error              |
//+------------------------------------------------------------------+
double RunModel(const double &input_matrix[])
{
   // Flatten to float32 array
   int cells = InpSeqLen * g_n_features;
   float in_buf[];
   ArrayResize(in_buf, cells);
   for(int i = 0; i < cells; i++)
      in_buf[i] = (float)input_matrix[i];

   float out_buf[];
   ArrayResize(out_buf, InpNAhead);

   if(!OnnxRun(g_onnx, ONNX_NO_CONVERSION, in_buf, out_buf))
   {
      Print("OnnxRun error: ", GetLastError());
      return -1.0;
   }
   return (double)out_buf[0];  // scaled prediction for step 1
}

//+------------------------------------------------------------------+
//| Check whether we already have an open position (any direction)   |
//+------------------------------------------------------------------+
bool HasOpenPosition()
{
   for(int i = PositionsTotal() - 1; i >= 0; i--)
      if(PositionGetSymbol(i) == _Symbol &&
         PositionGetInteger(POSITION_MAGIC) == InpMagic)
         return true;
   return false;
}

//+------------------------------------------------------------------+
//| OnTick                                                           |
//+------------------------------------------------------------------+
void OnTick()
{
   // Only act once per new H1 bar
   datetime bar_time = iTime(_Symbol, PERIOD_H1, 0);
   if(bar_time == g_last_bar) return;
   g_last_bar = bar_time;

   //── 1. Gather ATR ────────────────────────────────────────────
   double atr_buf[];
   if(CopyBuffer(g_atr_handle, 0, 1, 1, atr_buf) < 1)
   {
      Print("ATR copy failed"); return;
   }
   double atr = atr_buf[0];
   if(atr <= 0.0) return;

   //── 2. Build combined history for all features ────────────────
   // Layout: [time × feature]  feature order = primary_close, extra0_close, extra1_close ...
   double raw_all[];
   ArrayResize(raw_all, g_hist_needed * g_n_features);

   // Primary symbol
   double raw0[];
   if(!GetCloseHistory(_Symbol, g_hist_needed, raw0)) return;
   for(int t = 0; t < g_hist_needed; t++)
      raw_all[t * g_n_features + 0] = raw0[t];

   // Extra symbols
   for(int s = 0; s < ExtraSymbolCount(); s++)
   {
      double rawS[];
      if(!GetCloseHistory(ExtraSymbol(s), g_hist_needed, rawS)) return;
      for(int t = 0; t < g_hist_needed; t++)
         raw_all[t * g_n_features + (s + 1)] = rawS[t];
   }

   //── 3. Rolling-median normalise each feature ──────────────────
   // We need [seq_len × n_features] normalised values.
   // Also capture the target symbol's last-bar median for de-normalisation.
   double median_target = 1.0;
   double seq_matrix[];           // [seq_len × n_features]
   ArrayResize(seq_matrix, InpSeqLen * g_n_features);

   for(int f = 0; f < g_n_features; f++)
   {
      // Extract single-feature column
      double col_raw[];
      ArrayResize(col_raw, g_hist_needed);
      for(int t = 0; t < g_hist_needed; t++)
         col_raw[t] = raw_all[t * g_n_features + f];

      double col_norm[];
      double col_median;
      if(!NormaliseWithRollingMedian(col_raw, col_norm, col_median)) return;

      if(f == 0) median_target = col_median;

      for(int t = 0; t < InpSeqLen; t++)
         seq_matrix[t * g_n_features + f] = col_norm[t];
   }

   //── 4. Apply MinMax scaling (per feature, across time axis) ──
   // x_scaled = (x - X_min[f]) / X_scale[f]
   for(int t = 0; t < InpSeqLen; t++)
      for(int f = 0; f < g_n_features; f++)
      {
         int idx = t * g_n_features + f;
         seq_matrix[idx] = (seq_matrix[idx] - g_x_min[f]) / g_x_scale[f];
      }

   //── 5. Run ONNX ───────────────────────────────────────────────
   double y_scaled = RunModel(seq_matrix);
   if(y_scaled < 0.0) return;

   //── 6. Inverse MinMax on output ──────────────────────────────
   double y_norm = y_scaled * InpYScale + InpYMin;

   //── 7. Multiply by current median → real-price forecast ──────
   double forecast_price = y_norm * median_target;

   //── 8. Decide direction ───────────────────────────────────────
   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double mid = (ask + bid) * 0.5;

   double move = forecast_price - mid;
   double min_move = InpMinMoveRatio * atr;

   PrintFormat("Bar=%s  mid=%.5f  forecast=%.5f  move=%.5f  ATR=%.5f",
               TimeToString(bar_time), mid, forecast_price, move, atr);

   if(HasOpenPosition()) return;   // one position at a time

   if(MathAbs(move) < min_move)
   {
      Print("Signal too weak, skip");
      return;
   }

   //── 9. Calculate SL / TP ──────────────────────────────────────
   double sl_dist = InpSL_ATR_Mult * atr;
   double sl, tp;

   if(move > 0.0)   // BUY
   {
      sl = ask - sl_dist;
      tp = (InpTP_ATR_Mult > 0.0) ? ask + InpTP_ATR_Mult * atr : forecast_price;
   }
   else              // SELL
   {
      sl = bid + sl_dist;
      tp = (InpTP_ATR_Mult > 0.0) ? bid - InpTP_ATR_Mult * atr : forecast_price;
   }

   // Normalise to tick size
   double tick = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
   sl = MathRound(sl / tick) * tick;
   tp = MathRound(tp / tick) * tick;

   //── 10. Place order ───────────────────────────────────────────
   string comment = StringFormat("TKAN fc=%.2f", forecast_price);
   if(move > 0.0)
      g_trade.Buy(InpLots, _Symbol, ask, sl, tp, comment);
   else
      g_trade.Sell(InpLots, _Symbol, bid, sl, tp, comment);
}
//+------------------------------------------------------------------+