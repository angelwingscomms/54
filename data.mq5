#property copyright "Simple"
#property version "1.00"
#property script_show_inputs

input int InpBarsToCopy = 2000;

void OnStart() {
   Print("Fetching BTCUSD data...");
   
   string filename = "data.csv";
   int fileHandle = FileOpen(filename, FILE_CSV|FILE_WRITE, ",");
   
   if(fileHandle == INVALID_HANDLE) {
      Print("Failed to open file: ", GetLastError());
      return;
   }
   
   FileWrite(fileHandle, "datetime,open,high,low,close");
   
   MqlRates rates[];
   ArraySetAsSeries(rates, true);
   
   int totalBars = iBars("BTCUSD", PERIOD_H1);
   Print("Total bars available: ", totalBars);
   
   if(totalBars <= 0) {
      Print("No bars available");
      FileClose(fileHandle);
      return;
   }
   
   int copyCount = MathMin(InpBarsToCopy, totalBars);
   int copied = CopyRates("BTCUSD", PERIOD_H1, 0, copyCount, rates);
   
   if(copied <= 0) {
      Print("Failed to copy rates: ", GetLastError());
      FileClose(fileHandle);
      return;
   }
   
   Print("Copied ", copied, " bars");
   
   for(int i = 0; i < copied; i++) {
      string dtStr = TimeToString(rates[i].time, TIME_DATE|TIME_MINUTES);
      StringReplace(dtStr, ".", "-");
      StringReplace(dtStr, ":", "-");
      
      string line = dtStr + "," +
                  DoubleToString(rates[i].open, 2) + "," +
                  DoubleToString(rates[i].high, 2) + "," +
                  DoubleToString(rates[i].low, 2) + "," +
                  DoubleToString(rates[i].close, 2);
      
      FileWrite(fileHandle, line);
   }
   
   FileClose(fileHandle);
   
   Print("Written ", copied, " rows to ", filename);
}