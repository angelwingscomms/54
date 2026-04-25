#property copyright "Simple"
#property version "1.00"
#property strict

#include "norm_params.mqh"

input double LotSize = 0.01;
input int NumCandles = 45;

datetime lastBar = 0;
long OnnxHandle = INVALID_HANDLE;

#resource "model.onnx" as uchar ExtModel[]

int OnInit() {
   OnnxHandle = OnnxCreateFromBuffer(ExtModel, ONNX_DEFAULT);
   if(OnnxHandle == INVALID_HANDLE) { Print("ONNX create failed: ", GetLastError()); return INIT_FAILED; }
   return INIT_SUCCEEDED;
}

void OnDeinit(const int reason) {
   if(OnnxHandle != INVALID_HANDLE) OnnxRelease(OnnxHandle);
}

void OnTick() {
   datetime barTime = iTime("BTCUSD", PERIOD_CURRENT, 0);
   if(barTime == lastBar) return;
   lastBar = barTime;
   RunModel();
}

void RunModel() {
   const string sym = "BTCUSD";
   matrixf x(NumCandles, 4);
   for(int i = 0; i < NumCandles; i++) {
      int bar = NumCandles - 1 - i;
      x[i, 0] = (float)iOpen(sym, PERIOD_CURRENT, bar);
      x[i, 1] = (float)iHigh(sym, PERIOD_CURRENT, bar);
      x[i, 2] = (float)iLow(sym, PERIOD_CURRENT, bar);
      x[i, 3] = (float)iClose(sym, PERIOD_CURRENT, bar);
   }

   for(int f = 0; f < 4; f++) {
      double range = NORM_MAX[f] - NORM_MIN[f];
      if(range < 1e-8) range = 1e-8;
      for(int i = 0; i < NumCandles; i++)
         x[i, f] = (float)((x[i, f] - NORM_MIN[f]) / range);
   }

   matrixf x3d = x.Reshape(1, NumCandles * 4);
   vectorf y(1);
   if(!OnnxRun(OnnxHandle, 0, x3d, y)) { Print("ONNX run failed: ", GetLastError()); return; }
   Trade((double)y[0]);
}

void Trade(double pred) {
   if(pred > 0.5) {
      if(PositionSelect(Symbol()) && PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_SELL)
         CloseTrade();
      if(!PositionSelect(Symbol())) {
         MqlTradeRequest req = {}; MqlTradeResult res = {};
         req.action = TRADE_ACTION_DEAL; req.symbol = Symbol();
         req.volume = LotSize; req.price = SymbolInfoDouble(Symbol(), SYMBOL_ASK);
         req.type = ORDER_TYPE_BUY; req.comment = "TKAN_BUY";
         if(OrderSend(req, res) && res.retcode == TRADE_RETCODE_DONE) Print("BUY");
      }
   } else if(pred < 0.5) {
      if(PositionSelect(Symbol()) && PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY)
         CloseTrade();
      if(!PositionSelect(Symbol())) {
         MqlTradeRequest req = {}; MqlTradeResult res = {};
         req.action = TRADE_ACTION_DEAL; req.symbol = Symbol();
         req.volume = LotSize; req.price = SymbolInfoDouble(Symbol(), SYMBOL_BID);
         req.type = ORDER_TYPE_SELL; req.comment = "TKAN_SELL";
         if(OrderSend(req, res) && res.retcode == TRADE_RETCODE_DONE) Print("SELL");
      }
   }
}

void CloseTrade() {
   if(!PositionSelect(Symbol())) return;
   int type = (int)PositionGetInteger(POSITION_TYPE);
   MqlTradeRequest req = {}; MqlTradeResult res = {};
   req.action = TRADE_ACTION_DEAL; req.symbol = Symbol();
   req.volume = PositionGetDouble(POSITION_VOLUME);
   req.price = (type == POSITION_TYPE_BUY) ? SymbolInfoDouble(Symbol(), SYMBOL_BID) : SymbolInfoDouble(Symbol(), SYMBOL_ASK);
   req.type = (type == POSITION_TYPE_BUY) ? ORDER_TYPE_SELL : ORDER_TYPE_BUY;
   req.comment = "TKAN_CLOSE";
   if(OrderSend(req, res) && res.retcode == TRADE_RETCODE_DONE) Print("CLOSED");
}


