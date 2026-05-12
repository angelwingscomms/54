const int    MODEL_SEQ_LEN      = 54;
const int    MODEL_N_AHEAD      = 1;
const int    MODEL_ROLLING_DAYS = 14;
const int    MODEL_N_FEATURES   = 15;
const int    MODEL_N_OUT        = 3;

const string MODEL_PRIMARY_SYMBOL = "XAUUSD";
const string MODEL_TARGET_MODE    = "range_close";
const string MODEL_TARGET_NAMES[3] = {"close", "up_range", "down_range"};

const string MODEL_SYMBOLS[15] = {"XAUUSD", "XAUUSD", "XAUUSD", "XAUUSD", "XAUUSD", "$USDX", "$USDX", "$USDX", "$USDX", "$USDX", "USDJPY", "USDJPY", "USDJPY", "USDJPY", "USDJPY"};
const string MODEL_FEATURE_NAMES[15] = {"open", "high", "low", "close", "tick_volume", "open", "high", "low", "close", "tick_volume", "open", "high", "low", "close", "tick_volume"};

const double MODEL_X_MIN[15]   = {0.9285953, 0.9315812, 0.92495942, 0.92857045, 0.021008404, 0.99292284, 0.99317604, 0.99259603, 0.99294305, 0.03125, 0.98659545, 0.98677963, 0.98658186, 0.98654485, 0.014285714};
const double MODEL_X_SCALE[15] = {0.1234653, 0.1209107, 0.12623823, 0.12350494, 70.728989, 0.014687717, 0.014464557, 0.015035272, 0.014708281, 38.96875, 0.021602333, 0.021746933, 0.020879805, 0.021591067, 13.763492};

const double MODEL_Y_MIN[3]   = {0.92857045, 0, 0};
const double MODEL_Y_SCALE[3] = {0.12350494, 0.014326531, 0.023357252};
