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

using CommDataMap_t = std::unordered_map<std::tuple<int, int, int>, CommData, std::hash<std::tuple<int, int, int>>>;
using CommDataMapIter_t = typename CommDataMap_t::iterator;

class KokkosManager : public CBase_KokkosManager {
public:
  KokkosManager(CkMigrateMessage *msg) : CBase_KokkosManager(msg) {
    Kokkos::initialize();
  }

  ~KokkosManager() {
    Kokkos::finalize();
  }
};

class DomainChare : public CBase_DomainChare {
public:
  DomainChare(CkMigrateMessage *msg) : CBase_DomainChare(msg) {}

  DomainChare(int numRanks, Index_t nx_, int nr_,
              int balance_, int cost_, int showProg_, int quiet_,
              int its_, int viz_, int do_atomic_, 
              int numChares_);

  ~DomainChare() { delete locDom; }

  void init(int numRanks, Index_t nx, int tp, int nr,
            int balance, int cost, int numChares_);

  void CommDataInit(Domain& domain, Index_t dx, Index_t dy, Index_t dz, 
    bool doSend, bool planeOnly, CommDataMap_t &commDataMap);

  void CommSend(Domain& domain, int msgType, Index_t xferFields, 
    Kokkos::View<Real_t*> *fieldData, Index_t dx, Index_t dy, Index_t dz, 
    bool doSend, bool planeOnly, CommDataMap_t& commDataMap);

  void CommRecv(int& ref, int& x, int& y, int& z, int& xferFields, int& size, 
    Real_t* &buf, CkDeviceBufferPost* post);

  Real_t TimeStepCalculateLocal(Domain &domain);

  void TimeIncrement(Real_t newdt);

  void packingDone(int msgType, int x, int y, int z, int xferFields, 
    int sendCount, int offset);

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

class Main : public CBase_Main {
public:
  Main(CkArgMsg *m);
};

class PackingDoneMsg {
public:
  PackingDoneMsg() {}

  PackingDoneMsg(CProxy_DomainChare proxy_, int msgType_, int x_, int y_, int z_,
                int xferFields_, int sendCount_, int offset_)
      : proxy(proxy_), msgType(msgType_), x(x_), y(y_), z(z_),
        xferFields(xferFields_), sendCount(sendCount_), offset(offset_) {}

  CProxy_DomainChare proxy;
  int msgType;
  int x, y, z;
  int xferFields;
  int sendCount, offset;
};

#endif // LULESH_H
