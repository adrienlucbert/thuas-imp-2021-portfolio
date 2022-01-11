from pipetorch import flight_passengers
from pipetorch.train import *
from pipetorch.data import PTDataFrame
import time
from datetime import datetime, timedelta
from sklearn.metrics import *
import json
from DataFrameLoader import *
import sys

# Load configuration
config = None
with open(sys.argv[1]) as config_file:
    config = json.load(config_file)

# Set random state
torch.manual_seed(config["random_state"])
np.random.seed(config["random_state"])

# Load data
def factoryzero_date_parser(df: pd.DataFrame) -> pd.DataFrame:
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], unit='s').round('min')
    df = df.set_index("Timestamp")
    return df

filename = config["dataset_path"]
dfloader = DataFrameLoader.from_file(filename, date_parser=factoryzero_date_parser)
dfloader.add_targets('flow_temp', sheet_name='alklimaHeatPump')
dfloader.add_features('return_temp', sheet_name='alklimaHeatPump')
dfloader.add_features('power', sheet_name='energyHeatpump')
dfloader.add_features('flow_temp', 'return_temp', sheet_name='flowHeatSpaceHeating')

time_column_name = "Timestamp"
dfloader.add_index_as_feature(time_column_name)

df = dfloader.to_ptdataframe()
df = df.reset_index(drop=True)

def roundTime(dt=None, roundTo=100):
    #Converting numpy.datetime64 to datetime.datetime
    ts = pd.Timestamp(dt)
    dt = ts.to_pydatetime()
    seconds = (dt.replace(tzinfo=None) - dt.min).seconds
    rounding = (seconds + roundTo / 2) // roundTo * roundTo
    return dt + timedelta(0, rounding - seconds, -dt.microsecond)

df[time_column_name] = df[time_column_name].apply(roundTime)
df[time_column_name] = pd.to_numeric(df[time_column_name])

df = df.head(10000)
df = df.astype(np.float32).sequence(config["window_size"]).split(0.2).scale()
data = df.to_databunch(batch_size=config["batch_size"])

# Define the model
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size=100, num_layers=1, output_size=1, rnn=nn.LSTM):
        super().__init__()
        self.l1 = rnn(input_size, hidden_size, num_layers, batch_first=True)
        self.l2 = nn.Linear(hidden_size, output_size)

    def forward(self, X):
        h, _ = self.l1(X)
        h = h[:,-1, :]
        y = self.l2(h)
        y = y + X[:,-1,-1:]
        return y

rnns = {m.__name__: m for m in [nn.LSTM, nn.GRU]}
model = RNN(
    input_size=len(df.columns) - 1,
    hidden_size=config["hidden_size"],
    num_layers=config["num_layers"],
    output_size=1,
    rnn=rnns[config["rnn"]]
)

losses = {m.__name__: m for m in [nn.MSELoss, nn.HuberLoss]}
t = trainer(model, losses[config["loss"]](), data, metrics=r2_score, gpu=True)

# Perform the training
t.train(20, lr=(3e-4, 3e-2), report_frequency=1, save_lowest='loss')

# Calculate model fitness
print(f"Fitness: {t.evaluator.valid['r2_score'].max()}")