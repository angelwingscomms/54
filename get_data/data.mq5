#include "symbols.mq5"

input int InpBarsToCopy = 30000;
input ENUM_TIMEFRAMES InpTimeframe = PERIOD_H1;
input string InpOutputFile = "data.csv";

void OnStart() {
   Print("Fetching crypto OHLC data...");
   
   string filename = InpOutputFile;
   int fileHandle = FileOpen(filename, FILE_CSV|FILE_WRITE, ",");
   
   if(fileHandle == INVALID_HANDLE) {
      Print("Failed to open file: ", GetLastError());
      return;
   }
   
   FileWrite(fileHandle, "datetime", "symbol", "open", "high", "low", "close");

   int symbolCount = ArraySize(SYMBOLS);
   int totalRows = 0;

   for(int s = 0; s < symbolCount; s++) {
      string symbol = SYMBOLS[s];
      SymbolSelect(symbol, true);

      MqlRates rates[];
      ArraySetAsSeries(rates, true);

      int totalBars = iBars(symbol, InpTimeframe);
      if(totalBars <= 0) {
         Print("No bars available for ", symbol);
         continue;
      }

      int copyCount = MathMin(InpBarsToCopy, totalBars);
      int copied = CopyRates(symbol, InpTimeframe, 0, copyCount, rates);
      if(copied <= 0) {
         Print("Failed to copy rates for ", symbol, ": ", GetLastError());
         continue;
      }

      int digits = (int)SymbolInfoInteger(symbol, SYMBOL_DIGITS);
      for(int i = 0; i < copied; i++) {
         string dtStr = TimeToString(rates[i].time, TIME_DATE|TIME_MINUTES);
         StringReplace(dtStr, ".", "-");
         StringReplace(dtStr, ":", "-");

         FileWrite(
            fileHandle,
            dtStr,
            symbol,
            DoubleToString(rates[i].open, digits),
            DoubleToString(rates[i].high, digits),
            DoubleToString(rates[i].low, digits),
            DoubleToString(rates[i].close, digits)
         );
      }

      totalRows += copied;
      Print("Written ", copied, " rows for ", symbol);
   }
   
   FileClose(fileHandle);
   
   Print("Written ", totalRows, " rows to ", filename);
}
