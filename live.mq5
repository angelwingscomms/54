#property copyright "Simple"
#property version "1.00"
#property strict

#include "norm_params.mqh"
#include "config.mqh"

input double LotSize = 0.01;

datetime lastBar = 0;
long OnnxHandle = INVALID_HANDLE;

string gSymbol;

#include "model.onnx" as uchar ExtModel[]

int OnInit() {
   gSymbol = CFG_SYMBOL;
   if(StringLen(gSymbol) == 0) gSymbol = "BTCUSD";
   
   OnnxHandle = OnnxCreateFromBuffer(ExtModel, ONNX_DEFAULT);
   if(OnnxHandle == INVALID_HANDLE) { Print("ONNX create failed: ", GetLastError()); return INIT_FAILED; }
   return INIT_SUCCEEDED;
}

void OnDeinit(const int reason) {
   if(OnnxHandle != INVALID_HANDLE) OnnxRelease(OnnxHandle);
}

void OnTick() {
   datetime barTime = iTime(gSymbol, PERIOD_CURRENT, 0);
   if(barTime == lastBar) return;
   lastBar = barTime;
   RunModel();
}

void RunModel() {
   int numCandles = CFG_SEQUENCE_LENGTH;
   matrixf x(numCandles, 4);
   for(int i = 0; i < numCandles; i++) {
      int bar = numCandles - 1 - i;
      x[i, 0] = (float)iOpen(gSymbol, PERIOD_CURRENT, bar);
      x[i, 1] = (float)iHigh(gSymbol, PERIOD_CURRENT, bar);
      x[i, 2] = (float)iLow(gSymbol, PERIOD_CURRENT, bar);
      x[i, 3] = (float)iClose(gSymbol, PERIOD_CURRENT, bar);
   }

   for(int f = 0; f < 4; f++) {
      double range = NORM_MAX[f] - NORM_MIN[f];
      if(range < 1e-8) range = 1e-8;
      for(int i = 0; i < numCandles; i++)
         x[i, f] = (float)((x[i, f] - NORM_MIN[f]) / range);
   }

   vectorf y(1);
   matrixf x3d = x;
   x3d.Resize(1, 180);
   if(!OnnxRun(OnnxHandle, 0, x3d, y)) { Print("ONNX run failed: ", GetLastError()); return; }
   Trade((double)y[0]);
}

double GetSL(bool isBuy) {
   double entry = isBuy ? SymbolInfoDouble(gSymbol, SYMBOL_ASK) : SymbolInfoDouble(gSymbol, SYMBOL_BID);
   
   if(CFG_TARGET_TYPE == "atr") {
      double atr = iATR(gSymbol, PERIOD_CURRENT, CFG_ATR_PERIOD, 0);
      double slDist = atr * CFG_ATR_MULTIPLIER;
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
      double atr = iATR(gSymbol, PERIOD_CURRENT, CFG_ATR_PERIOD, 0);
      double slDist = atr * CFG_ATR_MULTIPLIER;
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