UNAME = $(shell uname)
ifeq ($(UNAME),Linux)
	CXX = g++-9
	LIB_DIR = /opt/intel/compilers_and_libraries/linux/lib/intel64
	LIBS = -pthread -lm -ldl
	MKL_ROOT = /opt/intel/compilers_and_libraries/linux/mkl
	MKL_LIB_DIR = $(MKL_ROOT)/lib/intel64
endif
ifeq ($(UNAME),Darwin)
	CXX = /usr/local/bin/g++-9
	LIB_DIR = /opt/intel/compilers_and_libraries/mac/lib
	MKL_ROOT = /opt/intel/compilers_and_libraries/mac/mkl
	MKL_LIB_DIR = $(MKL_ROOT)/lib
endif

CXXFLAGS = -m64 -fopenmp -O3

MKL_INC_DIR = $(MKL_ROOT)/include
LIBS = -lgomp -lm
# LIBS = -liomp5
MKL_LIBS = -lmkl_intel_lp64 -lmkl_sequential -lmkl_core 
# MKL_LIBS = -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core 

OSOBJS = Jacobi.o OneSidedJacobi.o
COBJS = Jacobi.o ClassicalJacobi.o
ROBJS = Jacobi.o CR_Jacobi.o
POBJS = Jacobi.o PO_Jacobi.o
OMPOBJS = Jacobi.o OmpPO_Jacobi.o

all:	CJacobi CRJacobi POJacobi OMPOJacobi OSJacobi

OSJacobi: $(OSOBJS)
	$(CXX) $(CXXFLAGS) -o OSJacobi $(OSOBJS) -L$(LIB_DIR) -L$(MKL_LIB_DIR) $(MKL_LIBS) $(LIBS)

OMPOJacobi: $(OMPOBJS)
	$(CXX) $(CXXFLAGS) -o OMPOJacobi $(OMPOBJS) -L$(LIB_DIR) -L$(MKL_LIB_DIR) $(MKL_LIBS) $(LIBS)

POJacobi: $(POBJS)
	$(CXX) $(CXXFLAGS) -o POJacobi $(POBJS) -L$(LIB_DIR) -L$(MKL_LIB_DIR) $(MKL_LIBS) $(LIBS)

CRJacobi: $(ROBJS)
	$(CXX) $(CXXFLAGS) -o CRJacobi $(ROBJS) -L$(LIB_DIR) -L$(MKL_LIB_DIR) $(MKL_LIBS) $(LIBS)

CJacobi: $(COBJS)
	$(CXX) $(CXXFLAGS) -o CJacobi $(COBJS) -L$(LIB_DIR) -L$(MKL_LIB_DIR) $(MKL_LIBS) $(LIBS)

%.o: %.cpp
	$(CXX) -c $(CXXFLAGS) -I$(MKL_INC_DIR) -o $@ $<

clean:
	rm -fr OSJacobi CJacobi CRJacobi POJacobi OMPOJacobi $(COBJS) $(ROBJS) $(POBJS) $(OMPOBJS) $(OSOBJS)
