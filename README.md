# Event-embedding-stock-prediction[WIP]

News titles and content were collected for top 10 companies from Nifty50, Nifty50 midcap and Nifty50 smallcap indices according to their market capitalisation ,from the following websites:
  * economictimes.indiatimes.com
  * reuters.com
  * ndtv.com
  * thehindubusinessline.com
  * moneycontrol.com
  * thehindu.com
  
However, only news titles were used for this experiment.
The summary of the news data collected from 1st Jan 2011 to 30th September 2017 is as follows:

|Nifty50 midcap cos.|No. of news|Nifty50 smallcap cos.|No. of news|Nifty 50 cos.|No. of news
| ------------- |:-------------:| -----:| -----:| -----:| -----:|
|indraprastha gas|329|indiabulls real estate|150|larsen & toubro|303
|federal bank|626|south indian bank|194|infosys|6751
|tata chemicals|331|indiabulls ventures|16|hdfc bank|1399
|voltas|335|escorts|290|reliance industries|10563
|page industries|56|future consumer|377|itc|2713
|divi's laboratories|337|care ratings|187|icici bank|3066
|mahindra & mahindra financial services|4997|equitas holdings|132|housing development finance corporation|3290
|l&t finance holdings|338|dcb bank|273|maruti suzuki india|1708
|bharat forge|351|can fin homes|108|tata consultancy services|321
|tvs motor company|520|sterlite technologies|761|kotak mahindra bank|891
|Total|8220|Total|2488|Total|31005

## Setup

The dataset can be downloaded from [here](https://drive.google.com/open?id=1GqNLoYnoAe4k2dgGg3eKLJCbqA6DH7zd).
	
## Prerequisites
  Following dependencies are required for this project:
  
    * en-vectors-web-lg>=2.0.0
    * h5py>=2.7.1
    * Keras>=2.1.1
    * numpy>=1.13.3
    * pandas>=0.21.0
    * pickleshare>=0.7.4
    * spacy>=2.0.2
    * tensorflow>=1.4.0
    * tensorflow-tensorboard>=0.4.0rc3
    
## File Structure 

	USEFUL_FILES
      -> vector-generation.py-Uses pre trained model from spacy to generate the vectors of news title.
      -> pre-processing.py-Rearranges the news data into three subgroups of short , medium and long 
                           term time period for each of the indices. It also aligns stock price date 
			 	with news date.
      -> model.py-Contains the architecture of the model used for training.
      
  
## Note

This model utilises the news data scraped from [this](https://github.com/gyanesh-m/Sentiment-analysis-of-financial-news-data) repo.


## Acknowledgement
This project was inspired from this [paper](https://www.ijcai.org/Proceedings/15/Papers/329.pdf)
