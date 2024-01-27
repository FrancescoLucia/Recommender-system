import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plotY(Y):
    plt.imshow(Y, cmap='viridis', aspect='auto')
    plt.ylabel('Movies')
    plt.xlabel('Users')
    plt.title('Movie Ratings Matrix')
    plt.colorbar(label='Rating')
    plt.show()


def normalizeRatings(Y, R):
    [m, n] = Y.shape
    Ymean = pd.Series(np.zeros(m))
    Ynorm = pd.DataFrame(np.zeros(Y.shape))
    for i in range(m):
        idx = np.where(R.iloc[i, :] == 1)[0]
        Ymean.iloc[i] = Y.iloc[i, idx].mean()
        Ynorm.iloc[i, idx] = Y.iloc[i, idx] - Ymean.iloc[i]
    return {"Ynorm": Ynorm, "Ymean": Ymean}


def costFunction(X, Theta, Y, R, usersNumber, filmsNumber, featuresNumber, lambda_):
    errors = pd.DataFrame((( np.dot(X.to_numpy(), Theta.T.to_numpy()) - Y.to_numpy()) * R.to_numpy()))
    squared_errors = errors ** 2
    J = ((1 / 2) * np.sum(squared_errors.to_numpy()) +
         (lambda_ / 2) * np.sum(Theta.to_numpy().flatten() ** 2) +
         (lambda_ / 2) * np.sum(X.to_numpy().flatten() ** 2))
    X_grad = np.dot(errors, Theta) + (lambda_ * X)
    Theta_grad = np.dot(errors.T, X) + (lambda_ * Theta)
    return [J, X_grad, Theta_grad]

def gradientDescent(X, Theta, Y, R, usersNumber, filmsNumber, featuresNumber, lambda_, alpha, iterations ):
    J_history = np.ones(iterations)
    for iter in range(iterations):
        [J, X_grad, Theta_grad] = costFunction(X, Theta, Y, R, usersNumber, filmsNumber, featuresNumber, lambda_)
        print(f"Iteration {iter}, cost: {J}")
        X -= alpha * X_grad
        Theta -= alpha * Theta_grad
        J_history[iter] = J
    return {"J_history": J_history, "X": X, "Theta": Theta}

def main():
    Y = pd.read_csv('Y.csv', sep=';', header=None)
    filmsNumber, usersNumber = Y.shape
    R = pd.read_csv('R.csv', sep=';', header=None)
    movieList = pd.read_csv("movie_ids.csv", sep=";")
    print(f"Loaded {filmsNumber} films and {usersNumber} users")

    filmId = 0
    filmName = movieList.iloc[filmId]["name"]
    filmRatings = Y.loc[filmId, R.iloc[filmId] == 1]
    filmMean = np.mean(filmRatings)
    print(f"Average rating for movie '{filmName}': {filmMean} / 5")

    plt.imshow(Y, cmap='viridis', aspect='auto')
    plt.ylabel('Movies')
    plt.xlabel('Users')
    plt.title('Movie Ratings Matrix')
    plt.colorbar(label='Rating')
    plt.show()

    newRatings = np.zeros(filmsNumber)
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
    print("New user ratings:")
    for index in range(filmsNumber):
        if newRatings[index] > 0:
            movieName = movieList.iloc[index]["name"]
            print(f"Rate {newRatings[index]} for film {movieName}")
    newRatingsSeries = pd.Series(newRatings)
    Y = pd.concat([newRatingsSeries, Y], axis=1)
    [filmsNumber, usersNumber] = Y.shape
    print(f"Now dataset contains {usersNumber} users")
    newRatingsBool = (newRatings != 0).astype(int)
    R = pd.concat([pd.Series(newRatingsBool), R], axis=1)

    normalizeResults = normalizeRatings(Y, R)
    Ynorm = normalizeResults["Ynorm"]
    Ymean = normalizeResults["Ymean"]

    featuresNumber = 20
    X = pd.DataFrame(np.random.randn(filmsNumber, featuresNumber))
    Theta = pd.DataFrame(np.random.randn(usersNumber, featuresNumber))
    lambda_ = 2
    alpha = 0.005
    iterations = 200
    trainingResults = gradientDescent(X, Theta, Ynorm, R, usersNumber, filmsNumber, featuresNumber, lambda_, alpha, iterations)
    J_history = trainingResults["J_history"]
    X = trainingResults["X"]
    Theta = trainingResults["Theta"]
    #plt.plot(J_history)
    #plt.show()
    print("Learning completed")

    print("Starting predictions")
    predictions = np.dot(X, Theta.T)
    newUserPrediction = pd.Series(predictions[:, 0] + Ymean)
    newUserPrediction = newUserPrediction.sort_values(ascending=False)
    for i in range(20):
        filmRate = newUserPrediction.iloc[i]
        filmId = newUserPrediction.index[i]
        filmName = movieList.iloc[filmId]["name"]
        print(f"Predicting rating {filmRate} for movie {filmName}")

    print("Finding similarities")
    movieId = 50
    print(f"Film: {movieList.iloc[movieId - 1]['name']}")
    similaritySeries = pd.Series(np.zeros(filmsNumber))
    for i in range(filmsNumber):
        similaritySeries.iloc[i] = np.linalg.norm(X.iloc[i, :] - X.iloc[movieId, :])
    similaritySeries.sort_values(ascending=True)
    max = 11
    preferedFilmsRate = similaritySeries.sort_values(ascending=True).iloc[1:max]
    for index in preferedFilmsRate.index:
        movieName = movieList.iloc[index]["name"]
        movieSimilarity = preferedFilmsRate.loc[index]
        print(f"Movie {movieName} --- with similarity {movieSimilarity}")

if __name__ == "__main__":
    main()
