#property copyright "Simple"
#property version "1.00"
#property strict

#include "norm_params.mqh"

input double LotSize = 0.01;
input int TimeframeMinutes = 60;
input int NumCandles = 45;
input int NumFeats = 40;

datetime lastBar = 0;
long OnnxHandle = INVALID_HANDLE;
ENUM_TIMEFRAMES TF;

string Symbols[] = {
   "BTCUSD", "ETHUSD", "ADAUSD", "EOSUSD", "MATICUSD",
   "TRXUSD", "XLMUSD", "LINKUSD", "BCHUSD", "LTCUSD"
};

#resource "model.onnx" as uchar ExtModel[]

int OnInit() {
   if(TimeframeMinutes < 1) { Print("TimeframeMinutes must be >= 1"); return INIT_FAILED; }
   switch(TimeframeMinutes) {
      case 1:  TF = PERIOD_M1;  break;
      case 5:  TF = PERIOD_M5;  break;
      case 15: TF = PERIOD_M15; break;
      case 30: TF = PERIOD_M30; break;
      case 60: TF = PERIOD_H1;  break;
      case 240: TF = PERIOD_H4; break;
      case 1440: TF = PERIOD_D1; break;
      default:  TF = PERIOD_CURRENT; break;
   }

OnnxHandle = OnnxCreateFromBuffer(ExtModel, ONNX_DEFAULT);
if(OnnxHandle == INVALID_HANDLE) { Print("ONNX create failed: ", GetLastError()); return INIT_FAILED; }
return INIT_SUCCEEDED;
}

void OnDeinit(const int reason) {
   if(OnnxHandle != INVALID_HANDLE) OnnxRelease(OnnxHandle);
}

void OnTick() {
   datetime barTime = iTime(Symbol(), TF, 0);
   if(barTime == lastBar) return;
   lastBar = barTime;
   RunModel();
}

void RunModel() {
   matrixf x(NumCandles, NumFeats);
   for(int s = 0; s < ArraySize(Symbols); s++) {
      string sym = Symbols[s];
      for(int i = 0; i < NumCandles; i++) {
         int bar = NumCandles - 1 - i;
         x[i, s*4+0] = (float)iOpen(sym, TF, bar);
         x[i, s*4+1] = (float)iHigh(sym, TF, bar);
         x[i, s*4+2] = (float)iLow(sym, TF, bar);
         x[i, s*4+3] = (float)iClose(sym, TF, bar);
      }
   }

   for(int f = 0; f < NumFeats; f++) {
      double range = NORM_MAX[f] - NORM_MIN[f];
      if(range < 1e-8) range = 1e-8;
      for(int i = 0; i < NumCandles; i++)
         x[i, f] = (float)((x[i, f] - NORM_MIN[f]) / range);
   }

   matrixf x3d[1];
   x3d[0] = x;
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


