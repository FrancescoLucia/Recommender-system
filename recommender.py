from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from Model import Movie
from MovieRecommender import MovieRecommender, plotJ

def loadData():
    Y = pd.read_csv('Y.csv', sep=';', header=None)
    R = pd.read_csv('R.csv', sep=';', header=None)
    movieDF = pd.read_csv("movie_ids.csv", sep=";")
    movieList = []
    for row in range(movieDF.shape[0]):
        movieId = movieDF.iloc[row]["id"]
        movieName = movieDF.iloc[row]["name"]
        movieList.append(Movie(movieId, movieName))
    return [Y, R, movieList]


def main():
    Y, R, movieList = loadData()

    recommender = MovieRecommender(Y, R, movieList)
    print(f"Loaded {recommender.moviesNumber} movies and {recommender.usersNumber} users")

    # Calculate ratings average for first film
    movieAverage = recommender.movieAverageRate(0)
    print(movieAverage)
    
    recommender.plotUsersRatings()

    # Add new ratings

    newRatings = np.zeros(recommender.moviesNumber)
    newRatings[0] = 4
    newRatings[97] = 2
    newRatings[49] = 5
    newRatings[55] = 4
    newRatings[66] = 4
    newRatings[71] = 4
    newRatings[81] = 5
    newRatings[95] = 4
    newRatings[120] = 4
    newRatings[126] = 5
    newRatings[143] = 5
    newRatings[173] = 5
    newRatings[179] = 4
    newRatings[187] = 5
    newRatings[209] = 5

    newRatingsList = recommender.addUserRatings(newRatings)
    print("New user ratings:")
    for movie in newRatingsList:
        print(movie)
    print(f"Now dataset contains {recommender.usersNumber} users")
    
    # Training
    featuresNumber = 20
    lambda_ = 2
    alpha = 0.002
    iterations = 200
    print("Start learning...")
    J_history = recommender.train(featuresNumber, lambda_, alpha, iterations)
    print("Learning completed")
    plotJ(J_history)

    predictedMovies = recommender.predict(userId=0, moviesNumber=20)
    for predictedMovie in predictedMovies:
        print(predictedMovie)

    print("Finding similarities")
    movieId = 50
    similarMovies = recommender.findSimilarMovies(movieId=movieId, moviesNumber=10)
    for movie in similarMovies:
        print(movie)
    
    print("-----")

if __name__ == "__main__":
    main()
