# 🛒 Market Basket Analysis using Apriori Algorithm

This project demonstrates how to perform **Market Basket Analysis** using the **Apriori Algorithm** and **Association Rule Mining** with the help of `mlxtend` and `pandas`.

---

## 📌 Description

Market Basket Analysis helps identify product purchase patterns from transactional data. This project uses a sample list of transactions to:

- Perform one-hot encoding of transaction data
- Find frequent itemsets using the Apriori algorithm
- Generate strong association rules (like "if milk, then bread")
- Sort and extract top rules based on lift metric

---

## 📂 Dataset

The dataset is a hard-coded list of transactions:

```python
transactions = [
    ['milk', 'bread', 'eggs'],
    ['milk', 'diapers', 'beer', 'bread'],
    ['milk', 'diapers', 'beer', 'cola'],
    ['milk', 'bread', 'diapers'],
    ['milk', 'bread', 'eggs'],
    ['milk', 'diapers', 'cola'],
    ['milk', 'bread', 'eggs'],
    ['milk', 'diapers', 'beer'],
]
##OUTPUT

---![Screenshot 2025-04-24 185645](https://github.com/user-attachments/assets/46b723d9-dc2f-45bc-aff1-9366455ba2e9)

![image](https://github.com/user-attachments/assets/cbc69e74-2380-4cb8-bc77-a5851cee34f0)

