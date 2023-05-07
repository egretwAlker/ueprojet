import json
import csv

def categories_description_pairs(fn):
  with open(fn) as f:
    data = [json.loads(s) for s in f.readlines()]
  return [(d['categories'].split('|'), d['description']) for d in data if len(d['categories'])>0]

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

def dedup(pairs):
  s = set()
  res = []
  for p in pairs:
    if p[1] not in s:
      res.append(p)
      s.add(p[1])
  return res

def essays(fn):
  res = []
  with open(fn, 'r', encoding='latin-1') as f:
    tsvreader = csv.reader(f, delimiter='\t')
    for row in tsvreader:
      if row[1] == '2':
        res.append((int(row[6]), row[2]))
  return res