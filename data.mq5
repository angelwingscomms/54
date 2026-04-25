#property copyright "Simple"
#property version "1.00"
#property script_show_inputs

input string StartDate = "2021.05.01";
input string EndDate = "2021.07.01";
input string SymbolName = "BTCUSD";

datetime startTime, endTime;

int OnInit() {
   startTime = StringToTime(StartDate);
   endTime = StringToTime(EndDate);
   
   Print("Fetching ", SymbolName, " data from ", StartDate, " to ", EndDate);
   
   fetchAndSaveData();
   
   return INIT_SUCCEEDED;
}

void fetchAndSaveData() {
   string filename = "data.csv";
   int fileHandle = FileOpen(filename, FILE_CSV|FILE_WRITE, ",");
   
   if(fileHandle == INVALID_HANDLE) {
      Print("Failed to open file: ", GetLastError());
      return;
   }
   
   FileWrite(fileHandle, "datetime,open,high,low,close,volume");
   
   MqlRates rates[];
   ArraySetAsSeries(rates, true);
   
   int copied = CopyRates(SymbolName, PERIOD_H1, startTime, endTime, rates);
   
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
                  DoubleToString(rates[i].close, 2) + "," +
                  DoubleToString(rates[i].tick_volume, 2);
      
      FileWrite(fileHandle, line);
   }
   
   FileClose(fileHandle);
   
   Print("Successfully written ", copied, " rows to ", filename);
}