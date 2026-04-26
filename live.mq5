#property copyright "Simple"
#property version "1.00"
#property strict

#include "models/well-2604-184235/config.mqh"
#include "models/well-2604-184235/norm_params.mqh"
#include "live/common.mq5"
#include "live/log_returns.mq5"
#include "live/candle_ratios.mq5"
#include "live/garman_klass.mq5"
#include "live/rolling_volatility.mq5"
#include "live/bollinger.mq5"
#include "live/price_to_sma.mq5"
#include "live/price_to_ema.mq5"
#include "live/ema_cross.mq5"
#include "live/rsi.mq5"
#include "live/adx.mq5"
#include "live/macd.mq5"
#include "live/higher_timeframes.mq5"
#include "live/time_features.mq5"

#resource "\\Experts\\54\\models\\well-2604-184235\\model.onnx" as uchar ExtModel[]

input double LotSize = 0.01;

datetime lastBar = 0;
long gOnnxHandle = INVALID_HANDLE;
int gAtrHandle = INVALID_HANDLE;

string gSymbol;
string gFeatureSymbols[];
int gFeatureCount = 0;

int OnInit() {
   if(Period() != PERIOD_M1) {
      Print("Attach live.mq5 to an M1 chart.");
      return INIT_FAILED;
   }
   gSymbol = CFG_SYMBOL;
   if(StringLen(gSymbol) == 0) gSymbol = _Symbol;
   SymbolSelect(gSymbol, true);
   gFeatureCount = StringSplit(CFG_FEATURE_SYMBOLS, ',', gFeatureSymbols);
   if(gFeatureCount <= 0) {
      ArrayResize(gFeatureSymbols, 1);
      gFeatureSymbols[0] = gSymbol;
      gFeatureCount = 1;
   }
   if(ArraySize(NORM_MIN) != CFG_INPUT_DIM || ArraySize(NORM_MAX) != CFG_INPUT_DIM) {
      Print(
         "NORM/config mismatch. Re-train and recompile live.mq5. CFG_INPUT_DIM=",
         CFG_INPUT_DIM,
         " NORM_MIN=",
         ArraySize(NORM_MIN),
         " NORM_MAX=",
         ArraySize(NORM_MAX)
      );
      return INIT_FAILED;
   }
   for(int i = 0; i < gFeatureCount; i++)
      SymbolSelect(gFeatureSymbols[i], true);

gOnnxHandle = OnnxCreateFromBuffer(ExtModel, ONNX_DEFAULT);
   if(gOnnxHandle == INVALID_HANDLE) { Print("ONNX create failed: ", GetLastError()); return INIT_FAILED; }
   
   if(CFG_TARGET_TYPE == "atr") {
      gAtrHandle = iATR(gSymbol, PERIOD_CURRENT, CFG_ATR_PERIOD);
      if(gAtrHandle == INVALID_HANDLE) { Print("ATR handle failed: ", GetLastError()); return INIT_FAILED; }
   }
   
   return INIT_SUCCEEDED;
}

bool BuildFeatureRow(const int barShift, double &row[])
{
   int offset = 0;
   for(int s = 0; s < gFeatureCount; s++) {
      string symbol = gFeatureSymbols[s];
      if(CFG_LOG_RETURNS_ENABLED && !AddLogReturnFeatures(symbol, PERIOD_CURRENT, barShift, row, offset)) return false;
      if(CFG_CANDLE_RATIOS_ENABLED && !AddCandleRatioFeatures(symbol, PERIOD_CURRENT, barShift, row, offset)) return false;
      if(CFG_GARMAN_KLASS_ENABLED && !AddGarmanKlassFeatures(symbol, PERIOD_CURRENT, barShift, row, offset)) return false;
      if(CFG_ROLLING_VOLATILITY_ENABLED && !AddRollingVolatilityFeatures(symbol, PERIOD_CURRENT, barShift, row, offset)) return false;
      if(CFG_BOLLINGER_ENABLED && !AddBollingerFeatures(symbol, PERIOD_CURRENT, barShift, row, offset)) return false;
      if(CFG_PRICE_TO_SMA_ENABLED && !AddPriceToSmaFeatures(symbol, PERIOD_CURRENT, barShift, row, offset)) return false;
      if(CFG_PRICE_TO_EMA_ENABLED && !AddPriceToEmaFeatures(symbol, PERIOD_CURRENT, barShift, row, offset)) return false;
      if(CFG_EMA_CROSS_ENABLED && !AddEmaCrossFeatures(symbol, PERIOD_CURRENT, barShift, row, offset)) return false;
      if(CFG_RSI_ENABLED && !AddRsiFeatures(symbol, PERIOD_CURRENT, barShift, row, offset)) return false;
      if(CFG_ADX_ENABLED && !AddAdxFeatures(symbol, PERIOD_CURRENT, barShift, row, offset)) return false;
      if(CFG_MACD_ENABLED && !AddMacdFeatures(symbol, PERIOD_CURRENT, barShift, row, offset)) return false;
      if(CFG_HIGHER_TIMEFRAMES_ENABLED && !AddHigherTimeframeFeatures(gSymbol, symbol, barShift, row, offset)) return false;
   }
   if(CFG_TIME_FEATURES_ENABLED && !AddTimeFeatures(gSymbol, barShift, row, offset)) return false;
   if(offset != CFG_INPUT_DIM) {
      Print("Feature count mismatch. Built=", offset, " expected=", CFG_INPUT_DIM);
      return false;
   }
   return true;
}

void OnDeinit(const int reason) {
   if(gOnnxHandle != INVALID_HANDLE) OnnxRelease(gOnnxHandle);
   if(gAtrHandle != INVALID_HANDLE) IndicatorRelease(gAtrHandle);
}

void OnTick() {
   datetime barTime = iTime(gSymbol, PERIOD_CURRENT, 0);
   if(barTime == 0) return;
   if(barTime == lastBar) return;
   Print("New bar detected: ", barTime);
   lastBar = barTime;
   RunModel();
}

void RunModel() {
   int numCandles = CFG_SEQUENCE_LENGTH;
   matrixf x(numCandles, CFG_INPUT_DIM);
   for(int i = 0; i < numCandles; i++) {
      int bar = numCandles - i;
      double row[];
      ArrayResize(row, CFG_INPUT_DIM);
      if(!BuildFeatureRow(bar, row)) {
         Print("Feature build failed at shift ", bar);
         return;
      }
      for(int col = 0; col < CFG_INPUT_DIM; col++)
         x[i, col] = (float)row[col];
   }

   for(int f = 0; f < CFG_INPUT_DIM; f++) {
      double range = NORM_MAX[f] - NORM_MIN[f];
      if(range < 1e-8) range = 1e-8;
      for(int i = 0; i < numCandles; i++)
         x[i, f] = (float)((x[i, f] - NORM_MIN[f]) / range);
   }

   vectorf y(1);
   matrixf x3d = x;
   x3d.Resize(1, numCandles * CFG_INPUT_DIM);
   if(!OnnxRun(gOnnxHandle, 0, x3d, y)) { Print("ONNX run failed: ", GetLastError()); return; }
   double buyProb = (double)y[0];
   if(!MathIsValidNumber(buyProb)) { Print("Invalid ONNX output: ", buyProb); return; }
   if(buyProb < 0.0 || buyProb > 1.0) { Print("ONNX output out of range: ", buyProb); return; }
   Print("buy_prob=", buyProb, " sell_prob=", 1.0 - buyProb);
   Trade(buyProb);
}

double GetSL(bool isBuy) {
   double entry = isBuy ? SymbolInfoDouble(gSymbol, SYMBOL_ASK) : SymbolInfoDouble(gSymbol, SYMBOL_BID);
   
   if(CFG_TARGET_TYPE == "atr") {
      double atrBuf[];
      CopyBuffer(gAtrHandle, 0, 0, 1, atrBuf);
      double slDist = atrBuf[0] * CFG_ATR_MULTIPLIER;
      return isBuy ? entry - slDist : entry + slDist;
   } else {
      double tpPct = CFG_THRESHOLD_PCT;
      double tol = CFG_TOLERANCE;
      return isBuy ? entry * (1 - tpPct * tol / 100) : entry * (1 + tpPct * tol / 100);
   }
}

double GetTP(bool isBuy) {
   double entry = isBuy ? SymbolInfoDouble(gSymbol, SYMBOL_ASK) : SymbolInfoDouble(gSymbol, SYMBOL_BID);
   
   if(CFG_TARGET_TYPE == "atr") {
      double atrBuf[];
      CopyBuffer(gAtrHandle, 0, 0, 1, atrBuf);
      double slDist = atrBuf[0] * CFG_ATR_MULTIPLIER;
      double tpDist = slDist * CFG_TP_MULTIPLIER;
      return isBuy ? entry + tpDist : entry - tpDist;
   } else {
      double tpPct = CFG_THRESHOLD_PCT;
      return isBuy ? entry * (1 + tpPct / 100) : entry * (1 - tpPct / 100);
   }
}

// Model output is buy probability: 1.0 = buy, 0.0 = sell.
void Trade(double buyProb) {
   double threshold = CFG_CONFIDENCE_THRESHOLD;
   double sl_price = 0, tp_price = 0;
   if(buyProb >= threshold) {
      if(PositionSelect(gSymbol) && PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_SELL)
         CloseTrade();
      if(!PositionSelect(gSymbol)) {
         sl_price = GetSL(true);
         tp_price = GetTP(true);
         MqlTradeRequest req = {}; MqlTradeResult res = {};
         req.action = TRADE_ACTION_DEAL; req.symbol = gSymbol;
         req.volume = LotSize; req.price = SymbolInfoDouble(gSymbol, SYMBOL_ASK);
         req.sl = sl_price; req.tp = tp_price;
         req.type = ORDER_TYPE_BUY; req.comment = "TKAN_BUY";
         if(OrderSend(req, res) && res.retcode == TRADE_RETCODE_DONE) Print("BUY buy_prob=", buyProb, " SL=", sl_price, " TP=", tp_price);
      }
   } else if(buyProb <= 1.0 - threshold) {
      if(PositionSelect(gSymbol) && PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY)
         CloseTrade();
      if(!PositionSelect(gSymbol)) {
         sl_price = GetSL(false);
         tp_price = GetTP(false);
         MqlTradeRequest req = {}; MqlTradeResult res = {};
         req.action = TRADE_ACTION_DEAL; req.symbol = gSymbol;
         req.volume = LotSize; req.price = SymbolInfoDouble(gSymbol, SYMBOL_BID);
         req.sl = sl_price; req.tp = tp_price;
         req.type = ORDER_TYPE_SELL; req.comment = "TKAN_SELL";
         if(OrderSend(req, res) && res.retcode == TRADE_RETCODE_DONE) Print("SELL buy_prob=", buyProb, " SL=", sl_price, " TP=", tp_price);
      }
   } else {
      Print("No trade. buy_prob=", buyProb, " threshold=", threshold);
   }
}

void CloseTrade() {
   if(!PositionSelect(gSymbol)) return;
   int type = (int)PositionGetInteger(POSITION_TYPE);
   MqlTradeRequest req = {}; MqlTradeResult res = {};
   req.action = TRADE_ACTION_DEAL; req.symbol = gSymbol;
   req.volume = PositionGetDouble(POSITION_VOLUME);
   req.price = (type == POSITION_TYPE_BUY) ? SymbolInfoDouble(gSymbol, SYMBOL_BID) : SymbolInfoDouble(gSymbol, SYMBOL_ASK);
   req.type = (type == POSITION_TYPE_BUY) ? ORDER_TYPE_SELL : ORDER_TYPE_BUY;
   req.comment = "TKAN_CLOSE";
   if(OrderSend(req, res) && res.retcode == TRADE_RETCODE_DONE) Print("CLOSED");
}
