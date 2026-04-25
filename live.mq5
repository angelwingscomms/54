#property copyright "Simple"
#property version "1.00"
#property strict

#include "config.mqh"
#include "norm_params.mqh"

#resource "\\Experts\\TKAN\\model.onnx" as uchar ExtModel[]

input double LotSize = 0.01;

datetime lastBar = 0;
long gOnnxHandle = INVALID_HANDLE;
int gAtrHandle = INVALID_HANDLE;

string gSymbol;
string gFeatureSymbols[];
int gFeatureCount = 0;

int OnInit() {
   gSymbol = CFG_SYMBOL;
   if(StringLen(gSymbol) == 0) gSymbol = _Symbol;
   SymbolSelect(gSymbol, true);
   gFeatureCount = StringSplit(CFG_FEATURE_SYMBOLS, ',', gFeatureSymbols);
   if(gFeatureCount <= 0) {
      ArrayResize(gFeatureSymbols, 1);
      gFeatureSymbols[0] = gSymbol;
      gFeatureCount = 1;
   }
   if(CFG_INPUT_DIM != gFeatureCount * 4) {
      Print("CFG_INPUT_DIM mismatch: expected ", gFeatureCount * 4, " got ", CFG_INPUT_DIM);
      return INIT_FAILED;
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
   double price0 = iClose(gSymbol, PERIOD_CURRENT, numCandles - 1);
   double price1 = iClose(gSymbol, PERIOD_CURRENT, 0);
   for(int i = 0; i < numCandles; i++) {
      int bar = numCandles - 1 - i;
      int col = 0;
      for(int s = 0; s < gFeatureCount; s++) {
         string symbol = gFeatureSymbols[s];
         double open = iOpen(symbol, PERIOD_CURRENT, bar);
         double high = iHigh(symbol, PERIOD_CURRENT, bar);
         double low = iLow(symbol, PERIOD_CURRENT, bar);
         double close = iClose(symbol, PERIOD_CURRENT, bar);
         if(open == 0.0 && high == 0.0 && low == 0.0 && close == 0.0) {
            Print("Missing bar data for ", symbol, " at shift ", bar);
            return;
         }
         x[i, col++] = (float)open;
         x[i, col++] = (float)high;
         x[i, col++] = (float)low;
         x[i, col++] = (float)close;
      }
   }

   for(int f = 0; f < CFG_INPUT_DIM; f++) {
      double range = NORM_MAX[f] - NORM_MIN[f];
      if(range < 1e-8) range = 1e-8;
      for(int i = 0; i < numCandles; i++)
         x[i, f] = (float)((x[i, f] - NORM_MIN[f]) / range);
   }

   Print("prices: oldest=", price0, " latest=", price1, " norm_min=", NORM_MIN[0], " norm_max=", NORM_MAX[0]);
   vectorf y(1);
   matrixf x3d = x;
   x3d.Resize(1, numCandles * CFG_INPUT_DIM);
   if(!OnnxRun(gOnnxHandle, 0, x3d, y)) { Print("ONNX run failed: ", GetLastError()); return; }
   Print("pred=", y[0]);
   Trade((double)y[0]);
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

void Trade(double pred) {
   double sl_price = 0, tp_price = 0;
   if(pred > 0.5) {
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
         if(OrderSend(req, res) && res.retcode == TRADE_RETCODE_DONE) Print("BUY SL=", sl_price, " TP=", tp_price);
      }
   } else if(pred < 0.5) {
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
         if(OrderSend(req, res) && res.retcode == TRADE_RETCODE_DONE) Print("SELL SL=", sl_price, " TP=", tp_price);
      }
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
