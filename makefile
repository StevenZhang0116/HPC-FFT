target=test

all_objs = $(basename $(wildcard *.cpp))

cpp_options= -std=c++11 -O3 -march=native -Wall -fopenmp
link_options= -fopenmp

target:${all_objs}

%:%.cpp
	g++ ${cpp_options} $< -o $@

clean:
	-$(RM) $(all_objs) *~

all:
	@make clean
	@make target

