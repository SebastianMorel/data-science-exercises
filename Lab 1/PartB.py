from bs4 import BeautifulSoup
from selenium import webdriver
import numpy as np
import pandas as pd
from pymongo import MongoClient
import matplotlib.pyplot as plt
import seaborn as sns
import requests

class PartB:
    def __init__(self, source=None, uri=None, collection=None):
        client = MongoClient('localhost',27017)
        db = client['lab1']
        self.col = db[collection]
        self._supported_data_sources = ('api', 'csv', 'mongodb', 'web')
        self._read_df(source, uri)
    
    def _read_df(self, source=None, uri=None):
        """
        Reads the data from a data source and converts it to a dataframe.
        
        Parameters
        ----------
        source : str
            Source of data source. Options: api, csv, mongodb, web
        uri : str
            Uri for data source. E.g., path for CSV, connection string for MongoDB, or URL for API and Web
        collection : str
            Collection to store the used data into the mongodb database.
        """
        if source is None:
            raise Exception('Error. No data source specified.')
    
        elif source not in self._supported_data_sources:
            raise Exception('Error. Unsupported data source specified.')

        if uri is None:
            raise Exception('Error. No URI was specified for the data source.')
            
        print(f'Data source: {source}')

        # Process API
        if source == 'api':
            r = requests.get(uri)
            self._df = pd.DataFrame(r.json()['Countries'])
            self.col.insert_many(self._df.to_dict('records'))
            
        # Process CSV
        elif source == 'csv':
            self._df = pd.read_csv(uri)
            self.col.insert_many(self._df.to_dict('records'))
        
        # Process MongoDB
        elif source == 'mongodb':
            c = MongoClient(uri)
            self._df = pd.DataFrame(list(c.lab1.covid.find()))
            self.col.insert_many(self._df.to_dict('records'))
        
        # Process Web
        elif source == 'web':
            driver = webdriver.Chrome('chromedriver.exe')
            driver.get(uri)
            html = driver.page_source
            self._soup = BeautifulSoup(html, 'html.parser')
            txtfile = {"file_name": "covid", "contents": self._soup.encode()}
            self.col.insert_one(txtfile)
        
    def plot_bargraph(self):
        """
        Format the data to process it in matplotlib, her we take the cases and deaths collumn values and group them into their respective WHO Region. 
        After this we create a list for each WHO Region that contains the mean value of cases and deaths for that WHO Region.
        """
        cases = ['Cases - cumulative total','Deaths - cumulative total']
        continents = self._df.groupby('WHO Region')[cases].mean()
        idx = np.arange(len(cases))
        Euro_mean = list(continents.T['Europe'])
        America_mean = list(continents.T['Americas'])
        EM_mean = list(continents.T['Eastern Mediterranean'])
        SEA_mean = list(continents.T['South-East Asia'])
        """
        Format the table and applying the data to the bars. Use style = 'plain' to not convert the y axis values to millions.
        """
        fig, ax = plt.subplots()
        plt.ticklabel_format(style = 'plain')
        bar_width = 0.20
        ax.bar(idx-bar_width, Euro_mean, bar_width, label='Europe')
        ax.bar(idx, America_mean, bar_width, label='America')
        ax.bar(idx+bar_width, EM_mean, bar_width, label='Eastern Mediterranean')
        ax.bar(idx+bar_width*2, SEA_mean, bar_width, label='South-East Asia')
        """
        Adding a legend to describe each color, and plotting the bar graph.
        """
        ax.set_xticks(idx)
        ax.set_xticklabels(cases)
        plt.title('Avg. Cases vs Avg. Deaths by regions')
        ax.legend()        
        plt.show()
        
    def plot_linechart(self):
        """
        Format the data to be able to process it in matplotlib, here we search the scraped page for column_name td which returns a list of the countries.
        Thenn we search for the "sc-fzqzEs hULauc" class which contains all the values but we only want cases so we take every four value which are the total cases.
        """
        countries = [x.text for x in self._soup.findAll("div", {"class": "column_name td"}, limit=10)]
        allmetrics = [x.text for x in self._soup.findAll("div", {"class": "sc-fzqzEs hULauc"}, limit=40)]
        allmetrics_clean = [int(x.replace(u'\xa0', u'')) for x in allmetrics]
        deaths = allmetrics_clean[::4]
        """
        We then plot the two lists into a line chart. Use set_ylin(bottom=0) to ensure that the y axis starts at the bottom of the chart to not mislead.
        We also resize and rotate the text on the x axis so the regions don't overlap
        """
        fig, ax = plt.subplots()
        plt.ticklabel_format(style = 'plain')
        plt.plot(countries,deaths)
        ax.set_ylim(bottom=0)
        plt.xticks(fontsize=8, rotation=90)
        plt.title('Total cases in the 10 most affected countries')
        plt.xlabel('Country')
        plt.ylabel('Cases')
        plt.show()
        
    def plot_heatmap(self):
        """
        We convert the values of the seventh column in the dataframe to a list, and we chose a number of rows that will when taken the root of return an integer to make the heatmap an even square.
        We then convert the list into an np array in order to reshape it into a square where it's sides are the lists length's square root.
        """
        DeathsperMillion = self._df.iloc[1:-12,7].to_list()
        assert np.sqrt(len(DeathsperMillion))%1 == 0
        DeathsperMillion = np.array(DeathsperMillion)
        DeathsperMillion_res = DeathsperMillion.reshape(int(np.sqrt(len(DeathsperMillion))),int(np.sqrt(len(DeathsperMillion))))
        """
        Here we create the heatmap with the help of seaborn that works well together with pyplot. We also remove the axis markers.
        """
        fig, ax = plt.subplots()
        sns.heatmap(DeathsperMillion_res, square=True, ax=ax)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.set_title("Distribution of Corona Deaths per million by Country")
        plt.show()
        
    def plot_scatter(self):
        """
        We create two lists from the fifth and seventh collumns in the dataset which are the cases and deaths respectively.
        """
        Cases = self._df.iloc[:,5].tolist()
        Deaths = self._df.iloc[:,7].tolist()
        """
        First we input the lists with the data into the scatter plots, then we make the start value (0) for both axis start at origo. Label the x and y axis and then plot the scatter plot.
        """
        plt.scatter(Cases,Deaths)
        plt.ylim(bottom=0)
        plt.xlim(xmin=0)
        plt.xlabel('Deaths')
        plt.ylabel('Cases')
        plt.title('Deaths vs Cases for every country')
        plt.show()
        
class Covid(PartB):
    def __init__(self, source=None, uri=None, collection=None):
        super().__init__(source, uri, collection)
        
if __name__ == '__main__':
    # Scatterplot for COVID-19 using data source API
    Covid('api', uri='https://api.covid19api.com/summary', collection='api').plot_scatter()

    # Heatmap for COVID-19 using data source CSV
    Covid('csv', uri='covid.csv', collection='csv').plot_heatmap()

    # Bargraph for COVID-19 using data source mongodb
    Covid('mongodb', uri='mongodb://localhost:27017/', collection='DB').plot_bargraph()

    # Linechart for COVID-19 using data source WEB
    Covid('web', uri='https://covid19.who.int/table', collection='web').plot_linechart()