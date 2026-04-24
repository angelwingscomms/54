// live.mq5 - Simple ONNX trading bot
#property copyright "Simple"
#property version   "1.00"
#property strict

#include <OnnxRuntime.mqh>

input double LotSize = 0.01;
input string OnnxFile = "model.onnx";
input int NumCandles = 45;

datetime lastBar = 0;

int OnnxHandle = INVALID_HANDLE;

int OnInit() {
   Print("Loading ONNX: ", OnnxFile);
   OnnxHandle = OnnxRuntime::Load(OnnxFile);
   if(OnnxHandle == INVALID_HANDLE) {
      Print("ERROR: Failed to load ONNX");
      return INIT_FAILED;
   }
   Print("ONNX loaded OK");
   return INIT_SUCCEEDED;
}

void OnDeinit(const int reason) {
   if(OnnxHandle != INVALID_HANDLE) {
      OnnxRuntime::Free(OnnxHandle);
   }
}

void OnTick() {
   if(Time[0] == lastBar) return;
   lastBar = Time[0];
   
   // Wait for hour bar
   if(iBarShift(NULL, PERIOD_H1, Time[0], true) == iBarShift(NULL, PERIOD_H1, Time[0], false)) {
      return;
   }
   
   RunModel();
}

void RunModel() {
   // Make input array [1, 45, 19]
   double input[];
   ArrayResize(input, NumCandles * 19);
   
   for(int i = 0; i < NumCandles; i++) {
      int bar = NumCandles - 1 - i;
      // OHLCV: open, high, low, close, volume * 3 + others
      input[i * 19 + 0] = Open[bar];
      input[i * 19 + 1] = High[bar];
      input[i * 19 + 2] = Low[bar];
      input[i * 19 + 3] = Close[bar];
      input[i * 19 + 4] = Volume[bar];
      // Fill rest with close
      for(int k = 5; k < 19; k++) {
         input[i * 19 + k] = Close[bar];
      }
   }
   
// Run ONNX - input shape [1, 45, 19] -> output [1, 1]
   double input[];
   ArrayResize(input, 1 * NumCandles * 19);
   
   for(int i = 0; i < NumCandles; i++) {
      int bar = NumCandles - 1 - i;
      input[i * 19 + 0] = Open[bar];
      input[i * 19 + 1] = High[bar];
      input[i * 19 + 2] = Low[bar];
      input[i * 19 + 3] = Close[bar];
      input[i * 19 + 4] = (double)Volume[bar];
      for(int k = 5; k < 19; k++) input[i * 19 + k] = Close[bar];
   }
   
   double output[];
   if(!OnnxRuntime::Run(OnnxHandle, input, output, 1, NumCandles, 19)) {
      Print("ONNX run failed");
      return;
   }
   
   // Get prediction
   double pred = output[0];
   Print("Prediction: ", pred);
   
   // Trade
   Trade(pred);
}

void Trade(double pred) {
   // Simple: if pred > 0.5 buy, if pred < 0.5 sell
   if(pred > 0.5) {
      // Buy
      if(PositionSelect(Symbol())) {
         if(PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_SELL) {
            CloseTrade();
         }
      }
      if(!PositionSelect(Symbol())) {
         double price = SymbolInfoDouble(Symbol(), SYMBOL_ASK);
         int ticket = OrderSend(Symbol(), OP_BUY, LotSize, price, 3, 0, 0, "ONNX_BUY");
         if(ticket > 0) Print("BUY ", ticket);
      }
   } else if(pred < 0.5) {
      // Sell
      if(PositionSelect(Symbol())) {
         if(PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY) {
            CloseTrade();
         }
      }
      if(!PositionSelect(Symbol())) {
         double price = SymbolInfoDouble(Symbol(), SYMBOL_BID);
         int ticket = OrderSend(Symbol(), OP_SELL, LotSize, price, 3, 0, 0, "ONNX_SELL");
         if(ticket > 0) Print("SELL ", ticket);
      }
   }
}

void CloseTrade() {
   if(PositionSelect(Symbol())) {
      int type = PositionGetInteger(POSITION_TYPE);
      double price = (type == POSITION_TYPE_BUY) ? SymbolInfoDouble(Symbol(), SYMBOL_BID) : SymbolInfoDouble(Symbol(), SYMBOL_ASK);
      int ticket = OrderSend(Symbol(), (type == POSITION_TYPE_BUY) ? OP_SELL : OP_BUY, PositionGetDouble(POSITION_VOLUME), price, 3, 0, 0, "CLOSE");
      if(ticket > 0) Print("CLOSED ", ticket);
   }
}