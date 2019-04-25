WITH first_tr AS (SELECT transactions.user_id, transactions.created_date, transactions.currency, transactions.amount
                  FROM (SELECT user_id, MIN(created_date) AS first_created
                        FROM transactions
                        GROUP BY user_id) AS first_transaction
                         INNER JOIN transactions ON transactions.created_date = first_transaction.first_created
                                                      AND transactions.user_id = first_transaction.user_id
                  WHERE transactions.type = 'CARD_PAYMENT'
                    AND transactions.state = 'COMPLETED'),
     convert_currency AS (SELECT currency,
                                 created_date,
                                 mn_date,
                                 user_id,
                                 amount,
                                 CASE
                                   WHEN currency = 'USD' THEN 1
                                   ELSE rate END * amount / POWER(10, exponent) AS amount_usd
                          FROM (SELECT user_id, created_date, currency, base_ccy, amount, MAX(ts) mn_date
                                FROM first_tr
                                       LEFT JOIN fx_rates ON fx_rates.ccy = currency
                                                               AND base_ccy = 'USD'
                                                               AND (ts <= created_date)
                                GROUP BY user_id, created_date, currency, amount, base_ccy) AS date_rates
                                 LEFT JOIN fx_rates ON mn_date = ts
                                                         AND ccy = currency
                                                         AND fx_rates.base_ccy = date_rates.base_ccy
                                 LEFT JOIN currency_details ON currency_details.ccy = currency)
SELECT user_id, amount_usd
FROM convert_currency
WHERE amount_usd >= 10
ORDER BY amount_usd DESC

