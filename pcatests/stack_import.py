from dataclasses import dataclass
from datetime import datetime
from typing import List
from xml.dom import minidom

now = datetime(2018, 5, 1, 0, 0, 0, 0)
import numpy as np


@dataclass
class Post:
    score: int
    creation_date: datetime
    view_count: int
    body_length: int
    answer_count: int
    comment_count: int
    favourite_count: int

    @property
    def age(self) -> int:
        return round((now - self.creation_date).total_seconds())

    @property
    def as_list(self) -> List:
        return [self.score, self.age, self.view_count, self.body_length,
                self.answer_count, self.comment_count, self.favourite_count]


inputfile = "/media/datenneu/se-simulator/raw/astronomy.stackexchange.com/Posts.xml"
xmldoc = minidom.parse(inputfile)
itemlist = xmldoc.getElementsByTagName("row")
rawdata = []
for s in itemlist:
    if s.attributes["PostTypeId"].value != "1":
        continue
    basictime = ".".join(s.attributes["CreationDate"].value.split(".")[:-1])  # get rid of decimal seconds
    favourite_count = int(s.attributes["FavoriteCount"].value) if "FavoriteCount" in s.attributes else 0
    post = Post(
        score=int(s.attributes["Score"].value),
        creation_date=datetime.strptime(basictime, "%Y-%m-%dT%H:%M:%S"),
        view_count=int(s.attributes["ViewCount"].value),
        body_length=len(s.attributes["Body"].value),
        answer_count=int(s.attributes["AnswerCount"].value),
        comment_count=int(s.attributes["CommentCount"].value),
        favourite_count=favourite_count,
    )
    rawdata.append(post.as_list)

data = np.array(rawdata)
print(data.dtype)
np.savetxt("data.txt", data,fmt="%d")
