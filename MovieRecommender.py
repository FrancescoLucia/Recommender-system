import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from Model import Movie, MovieAverage, MoviePrediction, MovieRate

class MovieRecommender:
    def __init__(self, Y: pd.DataFrame, R: pd.DataFrame, movieList: list[Movie]):
        self.Y = Y
        self.R = R
        self.movieList = movieList
        self.moviesNumber, self.usersNumber = Y.shape
        self.X = None
        self.Theta = None
        self.predictions = None
        self.Ymean = None

    def movieAverageRate(self, movieId:int) -> MovieAverage:
        movieName = self.movieList[movieId].id
        movieRatings = self.Y.loc[movieId, self.R.iloc[movieId] == 1]
        movieMean = np.mean(movieRatings)
        return MovieAverage(movieId, movieName, movieMean)

    def normalizeRatings(self, Y:pd.DataFrame, R: pd.DataFrame):
        [m, n] = Y.shape
        Ymean = pd.Series(np.zeros(m))
        Ynorm = pd.DataFrame(np.zeros(Y.shape))
        for i in range(m):
            idx = np.where(R.iloc[i, :] == 1)[0]
            Ymean.iloc[i] = Y.iloc[i, idx].mean()
            Ynorm.iloc[i, idx] = Y.iloc[i, idx] - Ymean.iloc[i]
        return {"Ynorm": Ynorm, "Ymean": Ymean}

    def addUserRatings(self, userRatings: np.array) -> list[MovieRate]:
        newRatingsList = []
        for index in range(self.moviesNumber):
            if userRatings[index] > 0:
                movieName = self.movieList[index].name
                movieRate = userRatings[index]
                newRatingsList.append(MovieRate(index, movieName, movieRate))
        newRatingsSeries = pd.Series(userRatings)
        self.Y = pd.concat([newRatingsSeries, self.Y], axis=1)
        newRatingsBool = (userRatings != 0).astype(int)
        self.R = pd.concat([pd.Series(newRatingsBool), self.R], axis=1)
        self.usersNumber += 1
        return newRatingsList

    def _initializeParameters(self, featuresNumber: int) -> None:
        self.X = pd.DataFrame(np.random.randn(self.moviesNumber, featuresNumber))
        self.Theta = pd.DataFrame(np.random.randn(self.usersNumber, featuresNumber))

    def _costFunction(self, X, Theta, Y, R, usersNumber: int, moviesNumber: int, featuresNumber:int, lambda_):
        errors = pd.DataFrame(((np.dot(X.to_numpy(), Theta.T.to_numpy()) - Y.to_numpy()) * R.to_numpy()))
        squared_errors = errors ** 2
        J = ((1 / 2) * np.sum(squared_errors.to_numpy()) +
             (lambda_ / 2) * np.sum(Theta.to_numpy().flatten() ** 2) +
             (lambda_ / 2) * np.sum(X.to_numpy().flatten() ** 2))
        X_grad = np.dot(errors, Theta) + (lambda_ * X)
        Theta_grad = np.dot(errors.T, X) + (lambda_ * Theta)
        return [J, X_grad, Theta_grad]

    def _gradientDescent(self, Y: pd.DataFrame, usersNumber:int , moviesNumber:int , featuresNumber:int, lambda_, alpha, iterations:int):
        J_history = np.ones(iterations)
        X = self.X
        Theta = self.Theta
        R = self.R.copy()
        for iter in range(iterations):
            [J, X_grad, Theta_grad] = self._costFunction(X, Theta, Y, R, usersNumber, moviesNumber, featuresNumber, lambda_)
            print(f"Iteration {iter}, cost: {J}")
            X -= alpha * X_grad
            Theta -= alpha * Theta_grad
            J_history[iter] = J
        self.X = X
        self.Theta = Theta
        return J_history


    def plotUsersRatings(self): # Plot Y
        plt.figure("Y")
        plt.imshow(self.Y, cmap='viridis', aspect='auto')
        plt.ylabel('Movies')
        plt.xlabel('Users')
        plt.title('Movie Ratings Matrix')
        plt.colorbar(label='Rating')
        plt.draw()
        plt.show(block=False)

    def train(self, featuresNumber, lambda_, alpha, iterations) -> np.array:
        normalizeResults = self.normalizeRatings(self.Y, self.R)
        Ynorm = normalizeResults["Ynorm"]
        self.Ymean = normalizeResults["Ymean"]
        self._initializeParameters(featuresNumber)
        J_history = self._gradientDescent(Ynorm, self.usersNumber, self.moviesNumber, featuresNumber, lambda_, alpha, iterations)
        return J_history

    def predict(self, userId:int, moviesNumber:int) -> list[MoviePrediction]:
        if self.predictions is None:
            self.predictions = np.dot(self.X, self.Theta.T)
        userPrediction = pd.Series(self.predictions[:, userId] + self.Ymean)
        userPrediction = userPrediction.sort_values(ascending=False)
        moviesPredicted = []
        for i in range(moviesNumber):
            movieRate = userPrediction.iloc[i]
            movieId = userPrediction.index[i]
            movieName = self.movieList[movieId].name
            moviesPredicted.append(MoviePrediction(movieId, movieName, rate=movieRate))
        return moviesPredicted

    def findSimilarMovies(self, movieId: int, moviesNumber: int) -> list[MoviePrediction]:
        similaritySeries = pd.Series(np.zeros(self.moviesNumber))
        for i in range(self.moviesNumber):
            similaritySeries.iloc[i] = np.linalg.norm(self.X.iloc[i, :] - self.X.iloc[movieId, :])
        similaritySeries.sort_values(ascending=True)

        similarMovies = similaritySeries.sort_values(ascending=True).iloc[1:moviesNumber]
        similarMoviesList = []
        for index in similarMovies.index:
            movieName = self.movieList[index].name
            movieSimilarity = similarMovies.loc[index]
            similarMoviesList.append(MoviePrediction(index, movieName, similarity=movieSimilarity))
        return similarMoviesList

def plotJ(J_history) -> None:
        plt.figure("j")
        plt.plot(J_history)
        plt.title('Cost Function History')
        plt.ylabel('Cost Function')
        plt.xlabel('Iterations')
        plt.draw()
        plt.show(block=False)