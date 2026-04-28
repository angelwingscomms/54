#property copyright "Simple"
#property version "1.00"
#property strict

#include "models/2804-054612/config.mqh"
#include "models/2804-054612/norm_params.mqh"

#resource "\\Experts\\54\\models\\2804-054612\\model.onnx" as uchar ExtModel[]

input double LotSize = 0.01;
input double MinDiffPercent = 0.15;

datetime lastBar = 0;
long gOnnxHandle = INVALID_HANDLE;

string gSymbol;
string gFeatureSymbols[];
int gFeatureCount = 0;

int OnInit() {
   if(Period() != PERIOD_M1) {
      Print("Attach r.mq5 to an M1 chart.");
      return INIT_FAILED;
   }
   if(MinDiffPercent <= 0 || MinDiffPercent > 10) {
      Print("MinDiffPercent must be > 0 and <= 10. Current: ", MinDiffPercent);
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
      Print("NORM/config mismatch. Re-train. CFG_INPUT_DIM=", CFG_INPUT_DIM, " NORM_MIN=", ArraySize(NORM_MIN));
      return INIT_FAILED;
   }
   for(int i = 0; i < gFeatureCount; i++)
      SymbolSelect(gFeatureSymbols[i], true);

   gOnnxHandle = OnnxCreateFromBuffer(ExtModel, ONNX_DEFAULT);
   if(gOnnxHandle == INVALID_HANDLE) { Print("ONNX create failed: ", GetLastError()); return INIT_FAILED; }

   return INIT_SUCCEEDED;
}

bool BuildFeatureRow(const int barShift, double &row[]) {
   int offset = 0;
   double closeVals[];
   for(int s = 0; s < gFeatureCount; s++) {
      string symbol = gFeatureSymbols[s];
      double scaler = iClose(symbol, PERIOD_CURRENT, barShift + CFG_SEQUENCE_LENGTH + 1);
      if(scaler <= 0) {
         double sum = 0, count = 0;
         for(int j = 1; j <= 24 * 14; j++) {
            double c = iClose(symbol, PERIOD_CURRENT, barShift + CFG_SEQUENCE_LENGTH + j);
            if(c > 0) { sum += c; count++; }
         }
         scaler = count > 0 ? sum / count : 1.0;
      }
      for(int i = 0; i < CFG_SEQUENCE_LENGTH; i++) {
         double close = iClose(symbol, PERIOD_CURRENT, barShift + i);
         row[offset++] = (close > 0 && scaler > 0) ? close / scaler : 1.0;
      }
   }
   return offset == CFG_INPUT_DIM;
}

void OnDeinit(const int reason) {
   if(gOnnxHandle != INVALID_HANDLE) OnnxRelease(gOnnxHandle);
}

void OnTick() {
   datetime barTime = iTime(gSymbol, PERIOD_CURRENT, 0);
   if(barTime == 0) return;
   if(barTime == lastBar) return;
   Print("New bar: ", barTime);
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
      if(!BuildFeatureRow(bar, row)) { Print("Feature build failed at shift ", bar); return; }
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
   
   double predictedRatio = (double)y[0];
   if(!MathIsValidNumber(predictedRatio)) { Print("Invalid ONNX output: ", predictedRatio); return; }
   
   double scaler = iClose(gSymbol, PERIOD_CURRENT, 1);
   if(scaler <= 0) {
      double sum = 0, count = 0;
      for(int j = 1; j <= 24 * 14; j++) {
         double c = iClose(gSymbol, PERIOD_CURRENT, j);
         if(c > 0) { sum += c; count++; }
      }
      scaler = count > 0 ? sum / count : 1.0;
   }
   
   double predictedPrice = predictedRatio * scaler;
   double currentPrice = iClose(gSymbol, PERIOD_CURRENT, 0);
   
Print("Predicted ratio: ", predictedRatio, " | Predicted price: ", predictedPrice, " | Current: ", currentPrice);
    
   double diff = MathAbs(predictedPrice - currentPrice);
   double diffPercent = (diff / currentPrice) * 100;
   if(diffPercent < MinDiffPercent) {
      Print("Diff ", diffPercent, "% below threshold ", MinDiffPercent, "% - skipping");
      return;
   }
   
   double direction = predictedPrice - currentPrice;
   Print("Direction: ", direction);
   Trade(predictedPrice, currentPrice);
}

void Trade(double predictedPrice, double currentPrice) {
   double diff = MathAbs(predictedPrice - currentPrice);
   double tpDistance = diff * 0.80;
   double slDistance = diff * 0.18;
   
   if(predictedPrice > currentPrice) {
      if(!PositionSelect(gSymbol)) {
         double ask = SymbolInfoDouble(gSymbol, SYMBOL_ASK);
         double sl = currentPrice - slDistance;
         double tp = currentPrice + tpDistance;
         MqlTradeRequest req = {};
         req.action = TRADE_ACTION_DEAL;
         req.symbol = gSymbol;
         req.volume = LotSize;
         req.price = ask;
         req.sl = sl;
         req.tp = tp;
         req.type = ORDER_TYPE_BUY;
         req.comment = "R_BUY";
         MqlTradeResult res = {};
         if(OrderSend(req, res) && res.retcode == TRADE_RETCODE_DONE)
            Print("BUY predicted > current | price=", ask, " | SL=", sl, " | TP=", tp);
      }
   } else if(predictedPrice < currentPrice) {
      if(!PositionSelect(gSymbol)) {
         double bid = SymbolInfoDouble(gSymbol, SYMBOL_BID);
         double sl = currentPrice + slDistance;
         double tp = currentPrice - tpDistance;
         MqlTradeRequest req = {};
         req.action = TRADE_ACTION_DEAL;
         req.symbol = gSymbol;
         req.volume = LotSize;
         req.price = bid;
         req.sl = sl;
         req.tp = tp;
         req.type = ORDER_TYPE_SELL;
         req.comment = "R_SELL";
         MqlTradeResult res = {};
         if(OrderSend(req, res) && res.retcode == TRADE_RETCODE_DONE)
            Print("SELL predicted < current | price=", bid, " | SL=", sl, " | TP=", tp);
      }
   }
}