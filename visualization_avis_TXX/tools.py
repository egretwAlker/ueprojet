# bsondump avis.bson > avis.json
import json

def getNote(d):
  for (x, y) in d.items():
    return float(y)
  assert(False)

def note_comment_pairs(fn):
  with open(fn) as f:
    data = [json.loads(s) for s in f.readlines()]
  return [(getNote(d['note']), d['comment']) for d in data if 'comment' in d and 'note' in d and len(d['comment']) > 0]

def note_short_pairs(fn):
  with open(fn) as f:
    data = [json.loads(s) for s in f.readlines()]
  return [(getNote(d['note']), d['title_review']) for d in data if 'comment' in d and 'note' in d and len(d['title_review']) > 0]

def test():
  res = note_comment_pairs("avis.json")
  for (x, y) in res:
    print(type(x), x)
  # print('\n'.join(str(d) for d in res[:100]))

if __name__ == "__main__":
  test()