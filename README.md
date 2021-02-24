# cfm_data_challenge_ens
*Where will the next trade take place?*

# Context of the challenge
Most financial markets use an electronic trading mechanism called a limit order book to facilitate the trading of assets (stocks, futures, options, etc.). Participants submit (or cancel) orders to this electronic order book. These orders are requests to buy or sell a given quantity of an asset at a specified price, thus allowing buyers to be matched with sellers at a mutually agreed price. Since an asset can be traded on multiple trading venues, participants can choose to which venue they send an order. For instance, a US stock can be traded on various exchanges, such as NYSE, NASDAQ, Direct Edge or BATS. When sending an order, participants generally select the best available trading venue at that time. Their decisions may include a statistical analysis of past venue activity.

**Given recent trades and order books from a set of trading venues, predict on which trading venue the next trade will be executed.**

# Description of the data
For each row, we want predict on which venue the next trade will be executed. The stock is represented by a randomized stock_id and the day by a randomized day_id.

Each row provides a description of six order books, from six trading venues, and a history of ten trades for the corresponding asset.

## Order books
An order book lists the quantities of an asset that are currently on offer by sellers (who ask for higher prices) and the quantities that buyers wish to acquire (who bid at lower prices). The six order books (one for each trading venue) are described in the dataset through the best two bids and best two asks (which makes them respectively the two highest bid prices of the buyers and the two lowest ask prices of the sellers).

Each of the six books is described as follows:

* The `bid` column (respectively 'ask') represents the difference between the best bid (respectively best ask) and the aggregate mid-price, expressed in some fixed currency unit.

* The `bid1` column (respectively 'ask1') represents the difference between the second best bid (respectively second best ask) and the aggregate mid-price, expressed in some fixed currency unit.

* The `bid_size` column (respectively 'ask_size') represents the total number of stocks available at the best bid (respectively best ask) divided by the aggregate volume.

* The `bid_size1` column (respectively 'ask_size1') represents the total number of stocks available at the second best bid (respectively at the second best ask) divided by the aggregate volume.

The 'ts_last_update' column corresponds to the timestamp, given as a number of microseconds since midnight (local time), of the last update of the book.

## Trades
Each row also comprises a description of the ten last trades (ordered from the most recent one to the oldest one) for the corresponding asset. A trade represents a transaction of a certain quantity of an asset at a given price between a buyer and a seller. The ten trades from the history of trades are given are described as follows:

* Its quantity (`qty`): the number of stocks traded, divided by the aggregate volume (defined above in the section "Order book").

* Its timestamp (`tod`): when the trade was executed, given as a number of microseconds since midnight (local time).

* Its price (`price`), representing the difference between the trade price with the aggregate mid-price (defined in the section "Order book"), expressed in some fixed currency unit.

* Its source (`source_id`) representing the trading venue on which this particular trade was executed.

# Approach 
The approach I took in this challenge is rather classic. It consists in a in-depth feature engineering followed by a well-tuned XGBoost trained using GPUs.

## Preprocessing
Not a lot of preprocessing was needed since the data was rather clean. 
* I removed the stocks that were not in the test set
* I renamed columns from tuple to string to ease manipulation
* There were a few nans in the OB price. It means that no price is proposed at that point in time. To remove them, I replaced the price with a price very far from the mid. This was especially useful to make new features and avoid "propagating" nans
* Normalizing trade time w.r.t. the last trade in the list, so that trade times are comparable across samples
* Normalize last update of the orders books, to make them comparable

## Feature engineering
Feature engineering was the main part of the challenge. In order to predict where the next trade would go, I used a series of different features : 
### Features on last 10 trades
* Count the frequency of each venue among the last 3, 5 and 10 trades. This is quite an intuitive feature : if a venue was much used before, it will probably continue after. 
* Count the frequency of each venue among the trades that happened in the last 0.1, 1 and 10 seconds. This allow to capture only near trades, which makes them more relevant. 
### Features on order book size, price and update
* Compute the total book size for bid and ask and for both levels
* Rank OB by proposed price (one of the most powerful features). It allows to create a "stationary" feature that does not depend on the context/time of the day, to describe how attractive this order book is likely to be w.r.t others.
* 


