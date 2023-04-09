# bsondump avis.bson > avis.json
import json

def categories_description_pairs(fn):
  with open(fn) as f:
    data = [json.loads(s) for s in f.readlines()]
  return [(d['categories'], d['description']) for d in data]

def test():
  d = categories_description_pairs("details.json")
  print(len(d))
  print(d[:10])
  pass

if __name__ == "__main__":
  test()