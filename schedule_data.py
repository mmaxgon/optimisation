from dataclasses import dataclass
import numpy as np

############################################################################
# Data
############################################################################

T = 288  # 24 часа * 60 мин / 5 мин
period = range(T)

# Форматы фильмов
movie_formats = set(["A", "B"])

@dataclass
class Movie:
	len: int            # длительность в 5-минутных шагах
	min: int            # минимальное число в расписании
	max: int            # максимальное число в расписании
	format: str         # формат
	def __post_init__(self):
		assert(self.format in movie_formats)

np.random.seed(1)

# 20 фильмов: 10 формата A, 10 формата B
# Длительность: 1-3 часа (12-36 шагов)
movies = {}
for i in range(1, 21):
	fmt = "A" if i <= 10 else "B"
	length = np.random.randint(12, 37)
	min_shows = np.random.randint(1, 2)
	max_shows = np.random.randint(max(5, min_shows), 21)
	movies[f"movie_{i}"] = Movie(len=length, min=min_shows, max=max_shows, format=fmt)

movie_names = list(movies.keys())

@dataclass
class Hall:
	supported_formats: list[str]     # поддерживаемые форматы
	max_shows: int                              # максимальное число сеансов
	def __post_init__(self):
		assert(type(self.supported_formats) is list)
		assert(all([f in movie_formats for f in self.supported_formats]))

# 10 залов: 7 поддерживают оба формата, 2 только A, 1 только B
halls = {
	"hall_1": Hall(supported_formats=["A", "B"], max_shows=15),
	"hall_2": Hall(supported_formats=["A", "B"], max_shows=15),
	"hall_3": Hall(supported_formats=["A", "B"], max_shows=15),
	"hall_4": Hall(supported_formats=["A", "B"], max_shows=15),
	"hall_5": Hall(supported_formats=["A", "B"], max_shows=15),
	"hall_6": Hall(supported_formats=["A", "B"], max_shows=15),
	"hall_7": Hall(supported_formats=["A", "B"], max_shows=15),
	"hall_8": Hall(supported_formats=["A"], max_shows=15),
	"hall_9": Hall(supported_formats=["A"], max_shows=15),
	"hall_10": Hall(supported_formats=["B"], max_shows=15),
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

# Допустимые времена начала для каждого фильма (сеанс заканчивается не позднее T)
valid_starts = {m: range(T - movies[m].len + 1) for m in movies}

# Ожидаемые продажи: movie × start_time
sales = {m: [np.random.randint(low=1, high=10) for t in range(T)] for m in movies}
