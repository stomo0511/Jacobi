PROJECT_ROOT = $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

CXX = /usr/local/bin/g++-9
CPPFLAGS = -fopenmp
LDFLAGS = -lgomp

COBJS = Jacobi.o ClassicalJacobi.o
ROBJS = Jacobi.o CR_Jacobi.o
POBJS = Jacobi.o PO_Jacobi.o
OMPOBJS = Jacobi.o OmpPO_Jacobi.o

# ifeq ($(BUILD_MODE),debug)
# 	CFLAGS += -g
# else ifeq ($(BUILD_MODE),run)
# 	CFLAGS += -O2
# else
# 	$(error Build mode $(BUILD_MODE) not supported by this Makefile)
# endif

all: CJacobi CRJacobi POJacobi OMPOJacobi

OMPOJacobi: $(OMPOBJS)
	$(CXX) -o $@ $^ $(LDFLAGS)

POJacobi: $(POBJS)
	$(CXX) -o $@ $^ $(LDFLAGS)

CRJacobi: $(ROBJS)
	$(CXX) -o $@ $^ $(LDFLAGS)

CJacobi: $(COBJS)
	$(CXX) -o $@ $^ $(LDFLAGS)

%.o:	$(PROJECT_ROOT)%.cpp
	$(CXX) -c $(CFLAGS) $(CXXFLAGS) $(CPPFLAGS) -o $@ $<

%.o:	$(PROJECT_ROOT)%.c
	$(CC) -c $(CFLAGS) $(CPPFLAGS) -o $@ $< 

clean:
	rm -fr CJacobi CRJacobi POJacobi OMPOJacobi $(COBJS) $(ROBJS) $(POBJS) $(OMPOBJS)
