import pandas as pd 

class Airport_efficiency:
    
    def __init__(self, df: pd.Series) -> None:
        self.df = df
    
    def airport_efficiency(self) -> list:
        eff_arr = []
        ports = [x for x in self.df.DEPSTN.unique()]
        
        for port in ports:
            pass

        return eff_arr  


if __name__ == "__main__":
    df = pd.read_csv('data/train.csv', nrows=10)
    instance = Airport_efficiency(df)
    instance.airport_efficiency()
