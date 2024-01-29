class Movie:
    def __init__(self, id, name):
        self.id = id
        self.name = name
    
    def __str__(self) -> str:
        return f"Movie: {self.name}"

class MovieAverage(Movie):
    def __init__(self, id, name, rateMean):
        super().__init__(id, name)
        self.rateMean = rateMean

    def __str__(self) -> str:
        return f"Movie: {self.name} average rating: {self.rateMean} / 5"

class MovieRate(Movie):
    def __init__(self, id, name, rate):
        super().__init__(id, name)
        self.rate = rate

    def __str__(self) -> str:
        return f"Rate {self.rate} for movie {self.name}"

class MoviePrediction(Movie):
    def __init__(self, id, name, rate=None, similarity=None):
        super().__init__(id, name)
        self.rate = rate
        self.similarity = similarity

    def __str__(self) -> str:
        if self.rate is not None:
            return f"Predicting rating {self.rate} for movie {self.name}"
        return f"Movie {self.name} --- with similarity {self.similarity}"
