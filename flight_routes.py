import pandas as pd 

class Flight_routes:
    def __init__(self, df: pd.Series) -> None:
        self.df = df

    def get_flight_routes(self) -> list:
        day_arr =[x for x in self.df.DATOP.unique()]
        print(day_arr)


if __name__ == "__main__":
    df = pd.read_csv('data/train.csv', nrows=10)
    instance = Flight_routes(df)
    instance.get_flight_routes()