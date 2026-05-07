const int    MODEL_SEQ_LEN      = 54;
const int    MODEL_N_AHEAD      = 1;
const int    MODEL_ROLLING_DAYS = 14;
const int    MODEL_N_FEATURES   = 19;
const int    MODEL_N_OUT        = 1;

const string MODEL_SYMBOLS[19] = {"BTCUSD", "BCHUSD", "ETHUSD", "LTCUSD", "XRPUSD", "ADAUSD", "AVAXUSD", "AXSUSD", "DOGEUSD", "DOTUSD", "EOSUSD", "FILUSD", "LINKUSD", "MATICUSD", "MIOTAUSD", "SOLUSD", "TRXUSD", "UNIUSD", "XLMUSD"};
const string MODEL_FEATURE_NAMES[19] = {"close", "close", "close", "close", "close", "close", "close", "close", "close", "close", "close", "close", "close", "close", "close", "close", "close", "close", "close"};

const double MODEL_X_MIN[19]   = {0.82048023, 0.66799343, 0.75364989, 0.72633392, 0.78944552, 0.64301229, 0.65303206, 0.63971251, 0.65864885, 0.65895677, 0.66862094, 0.59053499, 0.68944603, 0.62711072, 0.62008029, 0.74339336, 0.82815224, 0.63758749, 0.69452381};
const double MODEL_X_SCALE[19] = {0.30623782, 0.61703837, 0.71112198, 0.61597377, 2.9633918, 0.86293256, 0.61521089, 1.7983828, 0.75305676, 0.62473035, 0.84044456, 0.59924608, 0.71135443, 1.1274165, 0.6020121, 0.70809656, 0.30208093, 1.0055633, 1.3697038};

const double MODEL_Y_MIN[1]   = {0.82048023};
const double MODEL_Y_SCALE[1] = {0.30623782};
