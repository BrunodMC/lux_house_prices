# lux_house_prices

Small project intended to build a neural network capable of predicting Luxembourg property prices based on data scraped from the atHome.lu website.

Includes a web scraping script tailored specifically for extracting all entries posted in atHome.lu.

Includes also a pipeline for cleaning and preparing the data for use as a training set.

Currently the NN performance is abysmal, partly due to low number of examples (~4000 usable entries) and partly due to the large number of dimensions included, most of which are very sparse.

