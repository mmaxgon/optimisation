from dataclasses import dataclass
import numpy as np

############################################################################
# Data
############################################################################

T = 24
period = range(T)

# Форматы фильмов
movie_formats = set(["A", "B"])

@dataclass
class Movie:
	len: int            # длительность
	min: int            # минимальное число в расписании
	max: int            # максимальное число в расписании
	format: str         # формат
	def __post_init__(self):
		assert(self.format in movie_formats)

movies = {
	"movie_1": Movie(len=5, min=3, max=5, format="A"),
	"movie_2": Movie(len=3, min=4, max=10, format="A"),
	"movie_3": Movie(len=7, min=2, max=4, format="B"),
}
movie_names = list(movies.keys())

@dataclass
class Hall:
	supported_formats: list[str]     # поддерживаемые форматы
	max_shows: int                              # максимальное число сеансов
	def __post_init__(self):
		assert(type(self.supported_formats) is list)
		assert(all([f in movie_formats for f in self.supported_formats]))

halls = {
	"hall_1": Hall(supported_formats=["A"], max_shows=15),
	"hall_2": Hall(supported_formats=["A"], max_shows=12),
	"hall_3": Hall(supported_formats=["A", "B"], max_shows=10),
}
hall_names = list(halls.keys())

movie_halls = {movie : [hall for hall in halls if movies[movie].format in halls[hall].supported_formats] for movie in movies}
hall_movies = {hall : [movie for movie in movies if movies[movie].format in halls[hall].supported_formats] for hall in halls}

# Максимальное число сеансов фильма
movie_max_shows = {m: movies[m].max for m in movies}

# Максимальное число сеансов в зале
hall_max_shows = {
	h: min(
		halls[h].max_shows, 
		(T // (min(movies[m].len for m in hall_movies[h]) + 1)) + 1
	)
	for h in halls
}

# Максимальное число сеансов данного фильма в данном зале
hall_movie_max_shows = {}
for h in halls:
	for m in movies:
		hall_movie_max_shows[h, m] = 0 if h not in movie_halls[m] else min(
			movie_max_shows[m], 
			hall_max_shows[h],
			(T // (movies[m].len + 1)) + 1
		)

np.random.seed(1)
sales = {m: [np.random.randint(low=1, high=10) for t in range(T)] for m in movies}

