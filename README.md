# lux_house_prices

Small project intended to build a neural network capable of predicting Luxembourg property prices based on data scraped from the atHome.lu website.

Includes a web scraping script tailored specifically for extracting all entries posted in atHome.lu.

Includes also a pipeline for cleaning and preparing the data for use as a training set.

Currently the NN performance is abysmal, partly due to low number of examples (~4000 usable entries) and partly due to the large number of dimensions included, most of which are very sparse.

### Potential Improvements:
  - Most obviously, scraping more sites. However, this is likely to have diminishing returns since there's likely a lot of overlap between them.
  - Feature engineering to combine several miscellaneous sparse dimensions (e.g. household energy rating, open/closed parking spaces, pets allowed, etc) into a single numerical metric, a sort of misc desirability score.
  - Changing the location column from categorical to a numerical metric indicating its proximity/connectivity to major hubs (e.g. how easily does it connect to Lux city or Esch).
  - Simply removing several columns that might be too obscure and difficult to factor into the value of a property.
