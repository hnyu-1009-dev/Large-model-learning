import pandas as pd

dfjo = pd.DataFrame(
    dict(
        A=range(1, 4),
        B=range(4, 7),
        C=range(7, 10),
    ),
    columns=list("ABC"),
    index=list("xyz"),
)
print(dfjo)

dfjo.to_json("df.json", orient="columns")

# {"A":{"x":1,"y":2,"z":3},"B":{"x":4,"y":5,"z":6},"C":{"x":7,"y":8,"z":9}}

dfjo.to_json("df.json", orient="split")

# {"columns":["A","B","C"],"index":["x","y","z"],"data":[[1,4,7],[2,5,8],[3,6,9]]}

dfjo.to_json("df.json", orient="index")

# {"x":{"A":1,"B":4,"C":7},"y":{"A":2,"B":5,"C":8},"z":{"A":3,"B":6,"C":9}}

dfjo.to_json("df.json", orient="records")

# [{"A":1,"B":4,"C":7},{"A":2,"B":5,"C":8},{"A":3,"B":6,"C":9}]

dfjo.to_json("df.json", orient="records", lines=True)

# {"A":1,"B":4,"C":7}
# {"A":2,"B":5,"C":8}
# {"A":3,"B":6,"C":9}
