import pandas as pd 
from  datetime import datetime
class Flight_routes:
    def __init__(self, df: pd.Series) -> None:
        self.df = df

    def get_flight_routes(self) -> list:
        arr = []
        days = [x for x in self.df.DATOP.unique()]
        print(days)
        for day in days:
            day_arr = self.df.loc[self.df.DATOP == day]
            print(day_arr['AC'])


if __name__ == "__main__":
    df = pd.read_csv('data/train.csv', nrows=1000)
    instance = Flight_routes(df)
    instance.get_flight_routes()
