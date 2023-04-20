from flask import Flask, request
from flask_cors import CORS, cross_origin
import pandas as pd
import sys
import yfinance as yf
import math  
import mibian
import scipy
import json
from datetime import datetime
import numpy as np
import random
from random import gauss
from random import seed
from matplotlib import pyplot
from arch import arch_model

TRADING_DAYS_PER_YEAR = 250
	
def calc_iv(underlyingPrice, strikePrice, interestRate, daysToExpiration, optionPrice, isCall):
	if isCall: 
		c = mibian.BS([underlyingPrice, strikePrice, interestRate, daysToExpiration], callPrice=optionPrice)
		return c.impliedVolatility
	else:
		p = mibian.BS([underlyingPrice, strikePrice, interestRate, daysToExpiration], putPrice=optionPrice)
		return p.impliedVolatility

def calc_greeks(underlyingPrice, strikePrice, interestRate, daysToExpiration, optionPrice, isCall):
	if daysToExpiration < 1:
		daysToExpiration = 1
	if isCall: 
		iv = calc_iv(underlyingPrice, strikePrice, interestRate, daysToExpiration, optionPrice, True)
		c = mibian.BS([underlyingPrice, strikePrice, interestRate, daysToExpiration], volatility=iv)
		data = {}
		data['iv'] = iv
		data['theta'] = c.callTheta
		data['delta'] = c.callDelta
		data['rho'] = c.callRho
		data['vega'] = c.vega
		data['gamma'] = c.gamma
		return data
	else:
		iv = calc_iv(underlyingPrice, strikePrice, interestRate, daysToExpiration, optionPrice, False)
		p = mibian.BS([underlyingPrice, strikePrice, interestRate, daysToExpiration], volatility=iv)
		data = {}
		data['iv'] = iv
		data['theta'] = p.putTheta
		data['delta'] = p.putDelta
		data['rho'] = p.putRho
		data['vega'] = p.vega
		data['gamma'] = p.gamma
		return data
		
def calc_hv(historicData, days):
	sigma = historicData["Close"].head(n=int(days)).std()
	hv = sigma*math.sqrt(365/int(days))
	pctSigma = hv/historicData["Close"].median()*100
	return str(pctSigma)
	
def get_basic_fields(stock):
	try:
		historicData = stock.history(period='2d')
		underlyingPrice = get_underlying(historicData)
		priceJson = generate_price_json(underlyingPrice, historicData.tail(2)['Close'].iloc[0])
	
		data = {}
		data['price'] = priceJson
		info = stock.info
		if "bid" in info:
			data['bid'] = info['bid']
		if "ask" in info:
			data['ask'] = info['ask']
		if "shortName" in info:
			data['name'] = info['shortName']
		if "lastDividendValue" in info:
			data['dividend'] = info['lastDividendValue']
		if "volume" in info:
			data['volume'] = info['volume']
		if "fiftyTwoWeekHigh" in info:
			data['high'] = info['fiftyTwoWeekHigh']
		if "fiftyTwoWeekLow" in info:
			data['low'] = info['fiftyTwoWeekLow']
		if "dividendRate" in info:
			data['dividendRate'] = info['dividendRate']
		if "dayHigh" in info:
			data['dayHigh'] = info['dayHigh']
		if "dayLow" in info:
			data['dayLow'] = info['dayLow']
		monthExpDate = calc_30dte_date(stock)
		data['30dAtmIV'] = get_30d_atm_iv(stock, monthExpDate, underlyingPrice)
		return data
	except Exception:
		data = {}
		return data

def calc_30dte_date(stock):
	for expStr in stock.options:
		expDate = datetime.strptime(expStr, '%Y-%m-%d')
		currentDate = datetime.today()
		dte = (expDate - currentDate).days
		if dte >= 30:
			return expStr

	
def get_30d_atm_iv(stock, monthExpDate, underlyingPrice):
	try:
		calls = stock.option_chain(monthExpDate).calls
		calls.sort_values(by=['strike'])
		atmCall = calls[(calls.inTheMoney == True)].iloc[-1]
		expDate = datetime.strptime(monthExpDate, '%Y-%m-%d')
		currentDate = datetime.today()
		dte = (expDate - currentDate).days
		if dte < 1:
			dte = 1
		lastIV = calc_iv(underlyingPrice, atmCall['strike'], 1, dte, atmCall['lastPrice'], True)
		return lastIV
	except Exception as e:
		print(str(e))

def get_dividends(stock):
	dividends = stock.dividends
	dividends = dividends.sort_index(ascending=False)
	divDf = pd.DataFrame().append(dividends)
	return divDf

def get_splits(stock):
	splits = stock.splits
	splitDf = pd.DataFrame().append(splits)
	return splitDf
	
def generate_price_json(currentPrice, lastPrice):
	data = {}
	data['price'] = currentPrice
	data['diff'] = currentPrice - lastPrice
	data['pct'] = (currentPrice - lastPrice)/lastPrice * 100
	return data
	
def get_underlying(historicData):
	underlyingPrice = historicData.tail(1)['Close'].iloc[0]
	return underlyingPrice
	
def is_valid_option_type(optType):
	if optType != "call" and optType != "put":
		raise Exception("optType must be call or put")
	
def calc_exp_moves(currentPrice, vol, daysToExpiration):
	sigma = currentPrice*vol*(daysToExpiration/256)**0.5/100
	data = {}
	data['1sdLower'] = currentPrice - sigma
	data['1sdUpper'] = currentPrice + sigma
	data['2sdLower'] = currentPrice - 2*sigma
	data['2sdUpper'] = currentPrice + 2*sigma
	return data
	
def calc_opt_probabilities(underlyingPrice, strikePrice, optionPrice, daysToExpiration, iv, optType):
	sigma = underlyingPrice*iv*(daysToExpiration/256)**0.5/100
	data = {}
	if optType == 'call':
		data['otm'] = scipy.stats.norm(underlyingPrice,sigma).cdf(strikePrice)
		data['sellProfit'] = scipy.stats.norm(underlyingPrice,sigma).cdf(strikePrice + optionPrice)
		if data['sellProfit'] == 0 and strikePrice < underlyingPrice:
			data['sellProfit'] = 0.50
		data['buyProfit'] = 1 - data['sellProfit']
	else:
		data['otm'] = 1 - scipy.stats.norm(underlyingPrice,sigma).cdf(strikePrice)
		data['sellProfit'] = scipy.stats.norm(underlyingPrice,sigma).cdf(strikePrice - optionPrice)
		if data['sellProfit'] == 1 and underlyingPrice < strikePrice:
			data['sellProfit'] = 0.50
		data['buyProfit'] = 1 - data['sellProfit']
	return data

def calc_intr_extr_values(underlyingPrice, optionPrice, strikePrice, optType):
	data = {}
	if optType == 'call':
		if strikePrice >= underlyingPrice:
			data['ext'] = optionPrice
			data['intr'] = 0
		else: 
			data['intr'] = underlyingPrice - strikePrice
			data['ext'] = optionPrice - data['intr']
	else:
		if strikePrice <= underlyingPrice:
			data['ext'] = optionPrice
			data['intr'] = 0
		else: 
			data['intr'] = strikePrice - underlyingPrice
			data['ext'] = optionPrice - data['intr']
	return data
	
def run_simulations(dte, iv, initialPrice, strikePrice, optType, isStochasticVol):
	is_valid_option_type(optType)
	NUM_SIMS = 500
	stock = yf.Ticker("^VVIX")
	info = stock.info
	#volOfVol = info["bid"]*(1/TRADING_DAYS_PER_YEAR)**0.5
	volOfVol = 88*(1/TRADING_DAYS_PER_YEAR)**0.5
	iv = 28
	meanFinalIntrVal = 0
	# run a N trial monte carlo simulation of SPX stock price movements over the specified number of minutes.
	for i in range(1, NUM_SIMS):
		currentPrice = initialPrice
		currentIV = iv
		for day in range(0, dte):
			if isStochasticVol: 
				currentIV = np.random.normal(currentIV, volOfVol, 1)
				if(i == 1):
					print(currentIV)
				if currentIV <= 0:
					currentIV = 0
			
			sigma = initialPrice*currentIV/(100*TRADING_DAYS_PER_YEAR**0.5)
			# stock price movements roughly follow a lognormal distribution, so we must normalize our mean and st dev
			normalStd = math.sqrt(math.log(1 + (sigma/currentPrice)**2))
			normalMean = math.log(currentPrice) - normalStd**2 / 2
			currentPrice = np.random.lognormal(normalMean, normalStd)

		if(optType == "call" and currentPrice > strikePrice):
			finalIntrVal = currentPrice - strikePrice
			meanFinalIntrVal = meanFinalIntrVal + finalIntrVal
		elif(optType == "put" and currentPrice < strikePrice):
			finalIntrVal = strikePrice - currentPrice
			meanFinalIntrVal = meanFinalIntrVal + finalIntrVal
	return meanFinalIntrVal / NUM_SIMS
		
def train_garch_1_1(stockHistoricalData):	
	# seed pseudorandom number generator
	seed(1)
	# train over prior years' data
	trainingData = stockHistoricalData["Close"].head(n=TRADING_DAYS_PER_YEAR)
	model = arch_model(trainingData, vol='GARCH', p=1, q=1)
	# fit model
	return model.fit()
	
def train_arch(stockHistoricalData):	
	# seed pseudorandom number generator
	seed(1)
	# train over prior years' data
	trainingData = stockHistoricalData["Close"].head(n=TRADING_DAYS_PER_YEAR)
	model = arch_model(trainingData, vol='GARCH', p=1)
	# fit model
	return model.fit()
	
def run_g_arch(stockHistoricalData, dte, isGarch):
	modelFit = train_garch_1_1(stockHistoricalData)
	# forecast the test set
	yhat = modelFit.forecast(horizon=dte, reindex=False)
	underlyingPrice = get_underlying(stockHistoricalData)
	hIndx = 'h.' + str(dte)
	return yhat.variance.iloc[0][hIndx]
	
def calc_g_arch_val(ticker, optType, isGarch):
	is_valid_option_type(optType)
	stock = yf.Ticker(ticker)
	stockHistoricalData = stock.history(period=str(TRADING_DAYS_PER_YEAR) + "d")
	underlyingPrice = get_underlying(stockHistoricalData)
	strikePrice = float(request.args.get('strikePrice'))
	interestRate = float(request.args.get('riskFreeRate'))
	dte = int(request.args.get('dte'))
		
	data = {}
	
	garchVariance = run_g_arch(stockHistoricalData, dte, isGarch)
	
	garchVol = (garchVariance**0.5)/underlyingPrice*100
	annualizedGarchVol = garchVol*(TRADING_DAYS_PER_YEAR/dte)**0.5
	data['forecastedVol'] = annualizedGarchVol
	if(optType == "put"):
		data['forecastedPrice'] = mibian.BS([underlyingPrice, strikePrice, interestRate, dte], volatility=annualizedGarchVol).putPrice
	elif(optType == 'call'):
		data['forecastedPrice'] = mibian.BS([underlyingPrice, strikePrice, interestRate, dte], volatility=annualizedGarchVol).callPrice

	
	jsonData = json.dumps(data)
	return str(jsonData)

application = Flask(__name__)
cors = CORS(application)
application.config['CORS_HEADERS'] = 'Content-Type'

@application.route('/garch/value/<ticker>/<optType>')
def calc_garch_val(ticker, optType):
	return calc_g_arch_val(ticker, optType, True)
	
@application.route('/arch/value/<ticker>/<optType>')
def calc_arch_val(ticker, optType):
	return calc_g_arch_val(ticker, optType, False)

@application.route('/details/<ticker>/<optType>')
def calc_option_details(ticker, optType):
	is_valid_option_type(optType)
	stock = yf.Ticker(ticker)
	historicData = stock.history(period='100d')
	underlyingPrice = get_underlying(historicData)
	strikePrice = float(request.args.get('strikePrice'))
	interestRate = float(request.args.get('riskFreeRate'))
	dte = int(request.args.get('dte'))
	optionPrice = float(request.args.get('optionPrice'))
	data = calc_greeks(underlyingPrice, strikePrice, interestRate, dte, optionPrice, optType == 'call')
	data['expMoves'] = calc_exp_moves(underlyingPrice, data['iv'], dte)
	data['probabilities'] = calc_opt_probabilities(underlyingPrice, strikePrice, optionPrice, dte, data['iv'], optType)
	data['intrExtrVals'] = calc_intr_extr_values(underlyingPrice, optionPrice, strikePrice, optType)
	
	data['10dHv'] = calc_hv(historicData, 10)
	data['30dHv'] = calc_hv(historicData, 30)
	data['45dHv'] = calc_hv(historicData, 45)
	data['60dHv'] = calc_hv(historicData, 60)
	data['100dHv'] = calc_hv(historicData, 100)

	if optType == 'call':
		data['10dHvPrice'] = mibian.BS([underlyingPrice, strikePrice, interestRate, dte], volatility=data['10dHv']).callPrice
		data['30dHvPrice'] = mibian.BS([underlyingPrice, strikePrice, interestRate, dte], volatility=data['30dHv']).callPrice
		data['45dHvPrice'] = mibian.BS([underlyingPrice, strikePrice, interestRate, dte], volatility=data['45dHv']).callPrice
		data['60dHvPrice'] = mibian.BS([underlyingPrice, strikePrice, interestRate, dte], volatility=data['60dHv']).callPrice
		data['100dHvPrice'] = mibian.BS([underlyingPrice, strikePrice, interestRate, dte], volatility=data['100dHv']).callPrice

	else:
		data['10dHvPrice'] = mibian.BS([underlyingPrice, strikePrice, interestRate, dte], volatility=data['10dHv']).putPrice
		data['30dHvPrice'] = mibian.BS([underlyingPrice, strikePrice, interestRate, dte], volatility=data['30dHv']).putPrice
		data['45dHvPrice'] = mibian.BS([underlyingPrice, strikePrice, interestRate, dte], volatility=data['45dHv']).putPrice
		data['60dHvPrice'] = mibian.BS([underlyingPrice, strikePrice, interestRate, dte], volatility=data['60dHv']).putPrice
		data['100dHvPrice'] = mibian.BS([underlyingPrice, strikePrice, interestRate, dte], volatility=data['100dHv']).putPrice

	jsonData = json.dumps(data)
	return str(jsonData)
		
@application.route('/vrp/<ticker>/<optType>')
def calc_black_scholes_vrp(ticker, optType):
	is_valid_option_type(optType)
	stock = yf.Ticker(ticker)
	strikePrice = float(request.args.get('strikePrice'))
	interestRate = float(request.args.get('riskFreeRate'))
	dte = int(request.args.get('dte'))		
	optionPrice = float(request.args.get('optionPrice'))
	dteStr = str(dte) + "d"
	historicData = stock.history(period=dteStr)
	underlyingPrice = get_underlying(historicData)
		
	data = {}
		
	data['hv'] = calc_hv(historicData, dte)

	if optType == 'call':
		data['hvPrice'] = mibian.BS([underlyingPrice, strikePrice, interestRate, dte], volatility=data['hv']).callPrice

	else:
		data['hvPrice'] = mibian.BS([underlyingPrice, strikePrice, interestRate, dte], volatility=data['hv']).putPrice

	data['volRiskPremium'] = optionPrice - data['hvPrice']
	jsonData = json.dumps(data)
	return str(jsonData)
		
@application.route('/montecarlo/value/<ticker>/<optType>')
def calc_monte_carlo_value(ticker, optType):
	is_valid_option_type(optType)
	stock = yf.Ticker(ticker)
	strikePrice = float(request.args.get('strikePrice'))
	interestRate = float(request.args.get('riskFreeRate'))
	dte = int(request.args.get('dte'))
	dteStr = str(dte) + "d"
	historicData = stock.history(period=dteStr)
	underlyingPrice = get_underlying(historicData)

	expDate = calc_30dte_date(stock)
	iv = get_30d_atm_iv(stock, expDate, underlyingPrice)
	
	data = {}
	data["dynamicVolValue"] = run_simulations(dte, iv, underlyingPrice, strikePrice, optType, True)
	data["staticVolValue"] = run_simulations(dte, iv, underlyingPrice, strikePrice, optType, False)
		
	jsonData = json.dumps(data)
	return str(jsonData)
		
@application.route('/changes/<ticker>/<optType>')
def calc_opt_changes(ticker, optType):
	is_valid_option_type(optType)
	newUndlngPrice = float(request.args.get('newUndlngPrice'))
	newVol = float(request.args.get('newVol'))
	stock = yf.Ticker(ticker)

	strikePrice = float(request.args.get('strikePrice'))
	interestRate = 1.1
	expDateStr = request.args.get('expDate')
	expDate = datetime.strptime(expDateStr, '%Y-%m-%d')
	currentDate = datetime.today()
	dte = (expDate - currentDate).days
	if dte < 1:
		dte = 1
	data = {}
	if optType == "call":
		data['newPrice'] = mibian.BS([newUndlngPrice, strikePrice, interestRate, dte], volatility=newVol).callPrice
	else:
		data['newPrice'] = mibian.BS([newUndlngPrice, strikePrice, interestRate, dte], volatility=newVol).putPrice

	jsonData = json.dumps(data)
	return str(jsonData)
	
@application.route('/summary/short/<ticker>')
def get_short_summary(ticker):
	stock = yf.Ticker(ticker)
	data = get_basic_fields(stock)
	jsonData = json.dumps(data)
	return str(jsonData)
	
# Endpoint exists separately from /summary/short since it requires an additional call to yfinance api
@application.route('/summary/long/<ticker>')
def get_long_summary(ticker):
	stock = yf.Ticker(ticker)
	data = get_basic_fields(stock)
	info = stock.info
	if "sector" in info:
		data['sector'] = info['sector']
	if "forwardPE" in info:
		data['forwardPE'] = info['forwardPE']
	if "forwardEps" in info:
		data['forwardEps'] = info['forwardEps']
	if "trailingPE" in info:
		data['trailingPE'] = info['trailingPE']
	if "trailingEps" in info:
		data['trailingEps'] = info['trailingEps']
	if "pegRatio" in info:
		data['pegRatio'] = info['pegRatio']
	if "bookValue" in info:
		data['bookValue'] = info['bookValue']
	if "marketCap" in info:
		data['marketCap'] = info['marketCap']
	
	jsonData = json.dumps(data)
	return str(jsonData)

@application.route('/hv/<ticker>')
def calc_historic_vol(ticker):
	stock = yf.Ticker(ticker)
	days = request.args.get('days')
	daysStr = days + "d"
	historicData = stock.history(period=daysStr)
	return calc_hv(historicData, days)
	
@application.route('/options/<ticker>')
def get_options_chain(ticker):
	expDate = request.args.get('expDate')
	stock = yf.Ticker(ticker)
	options = stock.option_chain(expDate)
	callsDf = pd.DataFrame().append(options.calls)
	putsDf = pd.DataFrame().append(options.puts)
	optsDf = pd.merge(callsDf, putsDf, on=["strike"], suffixes=["_call", "_put"])
	optsDf.sort_values(by=['strike'])
	parsedJson = json.loads(optsDf.to_json(orient="records"))
	jsonData = json.dumps(parsedJson)
	return jsonData

@application.route('/iv/<optType>')
def calc_opt_iv(optType):
	is_valid_option_type(optType)
	underlyingPrice = float(request.args.get('underlyingPrice'))
	strikePrice = float(request.args.get('strikePrice'))
	interestRate = float(request.args.get('interestRate'))
	daysToExpiration = float(request.args.get('daysToExpiration'))
	optionPrice = float(request.args.get('optionPrice'))
	return str(calc_iv(underlyingPrice, strikePrice, interestRate, daysToExpiration, optionPrice, optType == 'call'))
	
@application.route('/greeks/<optType>')
def calc_opt_greeks(optType):
	is_valid_option_type(optType)
	underlyingPrice = float(request.args.get('underlyingPrice'))
	strikePrice = float(request.args.get('strikePrice'))
	interestRate = float(request.args.get('interestRate'))
	daysToExpiration = float(request.args.get('daysToExpiration'))
	optionPrice = float(request.args.get('optionPrice'))
	data = calc_greeks(underlyingPrice, strikePrice, interestRate, daysToExpiration, optionPrice, optType == 'call')
	jsonData = json.dumps(data)
	return str(jsonData)
	
@application.route('/prob_otm/<optType>')
def calc_prob_otm(optType):
	is_valid_option_type(optType)
	underlyingPrice = float(request.args.get('underlyingPrice'))
	strikePrice = float(request.args.get('strikePrice'))
	interestRate = float(request.args.get('interestRate'))
	daysToExpiration = float(request.args.get('daysToExpiration'))
	optionPrice = float(request.args.get('optionPrice'))
	iv = calc_iv(underlyingPrice, strikePrice, interestRate, daysToExpiration, optionPrice, optType == 'call')
	data = calc_opt_probabilities(underlyingPrice, strikePrice, optionPrice, daysToExpiration, iv, optType)
	jsonData = json.dumps(data)
	return str(jsonData)
	
@application.route('/extrinsic_val/<optType>')
def calc_extrinsic_value(optType):
	is_valid_option_type(optType)
	underlyingPrice = float(request.args.get('underlyingPrice'))
	strikePrice = float(request.args.get('strikePrice'))
	optionPrice = float(request.args.get('optionPrice'))
	data = calc_intr_extr_values(underlyingPrice, optionPrice, strikePrice, optType)
	jsonData = json.dumps(data)
	return str(jsonData)
	
@application.route('/dividends/<ticker>')
def get_dividends_for_ticker(ticker):
	stock = yf.Ticker(ticker)
	divDf = get_dividends(stock)
	parsedJson = json.loads(divDf.to_json(orient="records"))
	jsonData = json.dumps(parsedJson)
	return jsonData
	
@application.route('/splits/<ticker>')
def get_splits_for_ticker(ticker):
	stock = yf.Ticker(ticker)
	splitDf = get_splits(stock)
	parsedJson = json.loads(splitDf.to_json(orient="records"))
	jsonData = json.dumps(parsedJson)
	return jsonData

@application.route('/moves/<ticker>')
def get_expected_moves_for_ticker(ticker):
	daysToExpiration = request.args.get('dte')
	stock = yf.Ticker(ticker)
	daysStr = daysToExpiration + "d"
	currentPrice = get_underlying(stock.history(period=daysStr))
	expDate = calc_30dte_date(stock)
	vol = get_30d_atm_iv(stock, expDate, currentPrice)
	data = calc_exp_moves(currentPrice, vol, float(daysToExpiration))
	jsonData = json.dumps(data)
	return str(jsonData)
	
@application.route('/tickers/prices')
def get_multiple_tickers():
	tickers = yf.download("spy qqq dia msft aapl goog tsla amzn nflx pltr rckt nio gme amc tlt gld slv", period = "2d", interval = "1d", group_by = 'ticker')
	data = {}
	data['SPY'] = generate_price_json(tickers['SPY'].tail(2)['Close'].iloc[1], tickers['SPY'].tail(2)['Close'].iloc[0])
	data['QQQ'] = generate_price_json(tickers['QQQ'].tail(2)['Close'].iloc[1], tickers['QQQ'].tail(2)['Close'].iloc[0])
	data['DIA'] = generate_price_json(tickers['DIA'].tail(2)['Close'].iloc[1], tickers['DIA'].tail(2)['Close'].iloc[0])
	data['MSFT'] = generate_price_json(tickers['MSFT'].tail(2)['Close'].iloc[1], tickers['MSFT'].tail(2)['Close'].iloc[0])
	data['AAPL'] = generate_price_json(tickers['AAPL'].tail(2)['Close'].iloc[1], tickers['AAPL'].tail(2)['Close'].iloc[0])
	data['GOOG'] = generate_price_json(tickers['GOOG'].tail(2)['Close'].iloc[1], tickers['GOOG'].tail(2)['Close'].iloc[0])
	data['TSLA'] = generate_price_json(tickers['TSLA'].tail(2)['Close'].iloc[1], tickers['TSLA'].tail(2)['Close'].iloc[0])
	data['AMZN'] = generate_price_json(tickers['AMZN'].tail(2)['Close'].iloc[1], tickers['AMZN'].tail(2)['Close'].iloc[0])
	data['NFLX'] = generate_price_json(tickers['NFLX'].tail(2)['Close'].iloc[1], tickers['NFLX'].tail(2)['Close'].iloc[0])
	data['PLTR'] = generate_price_json(tickers['PLTR'].tail(2)['Close'].iloc[1], tickers['PLTR'].tail(2)['Close'].iloc[0])
	data['RCKT'] = generate_price_json(tickers['RCKT'].tail(2)['Close'].iloc[1], tickers['RCKT'].tail(2)['Close'].iloc[0])
	data['NIO'] = generate_price_json(tickers['NIO'].tail(2)['Close'].iloc[1], tickers['NIO'].tail(2)['Close'].iloc[0])
	data['GME'] = generate_price_json(tickers['GME'].tail(2)['Close'].iloc[1], tickers['GME'].tail(2)['Close'].iloc[0])
	data['AMC'] = generate_price_json(tickers['AMC'].tail(2)['Close'].iloc[1], tickers['AMC'].tail(2)['Close'].iloc[0])
	data['TLT'] = generate_price_json(tickers['TLT'].tail(2)['Close'].iloc[1], tickers['TLT'].tail(2)['Close'].iloc[0])
	data['GLD'] = generate_price_json(tickers['GLD'].tail(2)['Close'].iloc[1], tickers['GLD'].tail(2)['Close'].iloc[0])
	data['SLV'] = generate_price_json(tickers['SLV'].tail(2)['Close'].iloc[1], tickers['SLV'].tail(2)['Close'].iloc[0])
	jsonData = json.dumps(data)
	return str(jsonData)
		
@application.route('/tickers/correlations')
def get_correlations():
	pastDays = request.args.get('pastDays') + "d"
	tickers = yf.download("spy qqq dia msft aapl goog tsla amzn nflx pltr rckt nio gme amc tlt gld slv", period = pastDays, interval = "1d", group_by = 'ticker')
	df_pivot = tickers.loc[:, (slice(None), 'Close')].apply(pd.to_numeric)
	#df_pivot.columns = df_pivot.columns.droplevel()
	corr_df = df_pivot.corr()
	#reset symbol as index (rather than 0-X)
	#corr_df.head().reset_index()
	#del corr_df.index.name
	#jsonData = json.dumps(corr_df)
	return str(corr_df.to_json())
		
@application.route('/correlation/recent/<ticker>/<corrTicker>')
def get_recent_correlation(ticker, corrTicker):
	pastDays = request.args.get('pastDays') + "d"
	tickers = yf.download(ticker + " " + corrTicker, period = pastDays, interval = "1d", group_by = 'ticker')
	df_pivot = tickers.loc[:, (slice(None), 'Close')].apply(pd.to_numeric)
	corr_df = df_pivot.corr(method='pearson')
	print(tickers)
	return str(corr_df.iloc[1][0])
	
@application.route('/correlation/period/<ticker>/<corrTicker>')
def get_period_correlation(ticker, corrTicker):
	startDate = request.args.get('startDate')
	endDate = request.args.get('endDate')
	tickers = yf.download(ticker + " " + corrTicker, start=startDate, end=endDate, interval = "1d", group_by = 'ticker')
	df_pivot = tickers.loc[:, (slice(None), 'Close')].apply(pd.to_numeric)
	corr_df = df_pivot.corr(method='pearson')
	print(tickers)
	return str(corr_df.iloc[1][0])

@application.route('/expirations/<ticker>')
def get_expirations(ticker):
	stock = yf.Ticker(ticker)
	expirations = stock.options
	jsonData = json.dumps(expirations)
	return str(jsonData)

# run the app.
if __name__ == "__main__":
	# Setting debug to True enables debug output. This line should be
	# removed before deploying a production app.
	application.debug = True
	application.run()