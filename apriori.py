from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import pandas as pd


transactions=[
    ['milk', 'bread', 'eggs'],
    ['milk', 'diapers', 'beer', 'bread'],
    ['milk', 'diapers', 'beer', 'cola'],
    ['milk', 'bread', 'diapers'],
    ['milk', 'bread', 'eggs'],
    ['milk', 'diapers', 'cola'],
    ['milk', 'bread', 'eggs'],
    ['milk', 'diapers', 'beer'],
]


te=TransactionEncoder()
te_ary=te.fit(transactions).transform(transactions)
df=pd.DataFrame(te_ary, columns=te.columns_)

print("one hot encoding Dataframe")
print(df.head())

frequent_itemsets=apriori(df, min_support=0.4, use_colnames=True)
print("Frequent itemsets")

rules=association_rules(frequent_itemsets, metric='lift', min_threshold=1)
print("Association rules")
print(rules)

rules_sorted = rules.sort_values(by='lift', ascending=False)
rules_sorted = rules_sorted.drop_duplicates(subset=['antecedents', 'consequents']).reset_index(drop=True)

top20=rules_sorted.head(20)

print("Top 20 rules without duplicates")
print(top20[["antecedents", "consequents", "support", "confidence", "lift"]])
