#ifndef LULESH_H
#define LULESH_H

#include <unordered_map>
#include <tuple>
#include <functional>

// OpenMP will be compiled in if this flag is set to 1 AND the compiler beging
// used supports it (i.e. the _OPENMP symbol is defined)

//#include "lulesh-domain.h"
#include "lulesh.decl.h"
#include "hapi.h"

struct TupleHash {
    template <class T1, class T2, class T3>
    std::size_t operator()(const std::tuple<T1, T2, T3>& v) const {
        // Use a basic hash-combining strategy
        auto h1 = std::hash<T1>{}(std::get<0>(v));
        auto h2 = std::hash<T2>{}(std::get<1>(v));
        auto h3 = std::hash<T3>{}(std::get<2>(v));
        
        // A common way to combine hashes to avoid collisions
        return h1 ^ (h2 << 1) ^ (h3 << 2); 
    }
};

using CommDataMap_t = std::unordered_map<std::tuple<int, int, int>, CommData, TupleHash>;
using CommDataMapIter_t = CommDataMap_t::iterator;

class KokkosManager : public CBase_KokkosManager {
public:
  KokkosManager() : CBase_KokkosManager() {
    Kokkos::initialize();
  }

  KokkosManager(CkMigrateMessage *msg) : CBase_KokkosManager(msg) {
    Kokkos::initialize();
  }

  ~KokkosManager() {
    Kokkos::finalize();
  }
};

class DomainChare : public CBase_DomainChare {
  DomainChare_SDAG_CODE

public:
  DomainChare(CkMigrateMessage *msg) : CBase_DomainChare(msg) {}

  DomainChare(int numRanks, Index_t nx_, int nr_,
              int balance_, int cost_, int showProg_, int quiet_,
              int its_, int viz_, int do_atomic_, 
              int numChares_);

  ~DomainChare() { delete locDom; }

  void CommDataInit(Domain& domain, Index_t dx, Index_t dy, Index_t dz, 
    bool doSend, bool planeOnly, CommDataMap_t &commDataMap);

  void CommSend(Domain& domain, int msgType, Index_t xferFields, 
    Kokkos::View<Real_t*> *fieldData, Index_t dx, Index_t dy, Index_t dz, 
    bool doSend, bool planeOnly, CommDataMap_t& commDataMap);

  //void CommRecv(int& ref, int& x, int& y, int& z, int& xferFields, int& size, 
  //  Real_t* &buf, CkDeviceBufferPost* post);

  Real_t TimeStepCalculateLocal(Domain &domain);

  void TimeIncrement(Real_t newdt);

  void CommRecv(int ref, int x, int y, int z, int xferFields, int& size, Real_t* &buf, CkDeviceBufferPost* post);

  void packingDone(PackingDoneMsg* msg);

  void processRemotePosVel(int ref, int x, int y, int z, int xferFields, 
    int size, Real_t* buf);

  void processRemoteQ(int ref, int x, int y, int z, int xferFields, int size, Real_t* buf);

  void processRemoteMass(int ref, int x, int y, int z, int xferFields, int size, Real_t* buf);

  void processRemoteForce(int ref, int x, int y, int z, int xferFields, int size, Real_t* buf);

  Domain *locDom;
  int iter;
  int flatIndex;
  int commNbrs, remoteCount;
  int numChares;
  int recvRef;
  struct cmdLineOpts opts;

  CommDataMap_t commDataPosVel;
  CommDataMap_t commDataMonoQ;
  CommDataMap_t commDataSBN;

  hapiStream_t commStream, computeStream;
  ExecSpace commSpace, computeSpace;
};

class PackingDoneMsg : public CMessage_PackingDoneMsg {
public:
  int msgType;
  int x, y, z;
  int xferFields, sendCount, offset;

  PackingDoneMsg(int msgType_, int x_, int y_, int z_,
                 int xferFields_, int sendCount_, int offset_)
      : msgType(msgType_), x(x_), y(y_), z(z_),
        xferFields(xferFields_), sendCount(sendCount_), offset(offset_) {}
};

class Main : public CBase_Main {
  Main_SDAG_CODE

public:
  Main(CkArgMsg *m);

  double start;
};

#endif // LULESH_H
