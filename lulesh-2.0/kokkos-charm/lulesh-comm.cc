#include "lulesh.h"


#include <mpi.h>
#include <string.h>

/* Comm Routines */

#define ALLOW_UNPACKED_PLANE false
#define ALLOW_UNPACKED_ROW   false
#define ALLOW_UNPACKED_COL   false

/*
   There are coherence issues for packing and unpacking message
   buffers.  Ideally, you would like a lot of threads to 
   cooperate in the assembly/dissassembly of each message.
   To do that, each thread should really be operating in a
   different coherence zone.

   Let's assume we have three fields, f1 through f3, defined on
   a 61x61x61 cube.  If we want to send the block boundary
   information for each field to each neighbor processor across
   each cube face, then we have three cases for the
   memory layout/coherence of data on each of the six cube
   boundaries:

      (a) Two of the faces will be in contiguous memory blocks
      (b) Two of the faces will be comprised of pencils of
          contiguous memory.
      (c) Two of the faces will have large strides between
          every value living on the face.

   How do you pack and unpack this data in buffers to
   simultaneous achieve the best memory efficiency and
   the most thread independence?

   Do do you pack field f1 through f3 tighly to reduce message
   size?  Do you align each field on a cache coherence boundary
   within the message so that threads can pack and unpack each
   field independently?  For case (b), do you align each
   boundary pencil of each field separately?  This increases
   the message size, but could improve cache coherence so
   each pencil could be processed independently by a separate
   thread with no conflicts.

   Also, memory access for case (c) would best be done without
   going through the cache (the stride is so large it just causes
   a lot of useless cache evictions).  Is it worth creating
   a special case version of the packing algorithm that uses
   non-coherent load/store opcodes?
*/

/******************************************/


/* doRecv flag only works with regular block structure */
void CommRecv(Domain& domain, int msgType, Index_t xferFields,
              Index_t dx, Index_t dy, Index_t dz, bool doRecv, bool planeOnly) {

   // printf("Entering CommRecv\n") ;

   if (domain.numRanks() == 1)
      return ;

   /* post recieve buffers for all incoming messages */
   int myRank ;
   Index_t maxPlaneComm = xferFields * domain.maxPlaneSize() ;
   Index_t maxEdgeComm  = xferFields * domain.maxEdgeSize() ;
   Index_t pmsg = 0 ; /* plane comm msg */
   Index_t emsg = 0 ; /* edge comm msg */
   Index_t cmsg = 0 ; /* corner comm msg */
   MPI_Datatype baseType = ((sizeof(Real_t) == 4) ? MPI_FLOAT : MPI_DOUBLE) ;
   bool rowMin, rowMax, colMin, colMax, planeMin, planeMax ;

   /* assume communication to 6 neighbors by default */
   rowMin = rowMax = colMin = colMax = planeMin = planeMax = true ;

   if (domain.rowLoc() == 0) {
      rowMin = false ;
   }
   if (domain.rowLoc() == (domain.tp()-1)) {
      rowMax = false ;
   }
   if (domain.colLoc() == 0) {
      colMin = false ;
   }
   if (domain.colLoc() == (domain.tp()-1)) {
      colMax = false ;
   }
   if (domain.planeLoc() == 0) {
      planeMin = false ;
   }
   if (domain.planeLoc() == (domain.tp()-1)) {
      planeMax = false ;
   }

   for (Index_t i=0; i<26; ++i) {
      domain.recvRequest[i] = MPI_REQUEST_NULL ;
   }

   MPI_Comm_rank(MPI_COMM_WORLD, &myRank) ;

   /* post receives */

   /* receive data from neighboring domain faces */
   if (planeMin && doRecv) {
      /* contiguous memory */
      int fromRank = myRank - domain.tp()*domain.tp() ;
      int recvCount = dx * dy * xferFields ;
      MPI_Irecv(domain.commDataRecvView.data() + pmsg * maxPlaneComm,
                recvCount, baseType, fromRank, msgType,
                MPI_COMM_WORLD, &domain.recvRequest[pmsg]) ;
      ++pmsg ;
   }
   if (planeMax) {
      /* contiguous memory */
      int fromRank = myRank + domain.tp()*domain.tp() ;
      int recvCount = dx * dy * xferFields ;
      MPI_Irecv(domain.commDataRecvView.data() + pmsg * maxPlaneComm,
                recvCount, baseType, fromRank, msgType,
                MPI_COMM_WORLD, &domain.recvRequest[pmsg]) ;
      ++pmsg ;
   }
   if (rowMin && doRecv) {
      /* semi-contiguous memory */
      int fromRank = myRank - domain.tp() ;
      int recvCount = dx * dz * xferFields ;
      MPI_Irecv(domain.commDataRecvView.data() + pmsg * maxPlaneComm,
                recvCount, baseType, fromRank, msgType,
                MPI_COMM_WORLD, &domain.recvRequest[pmsg]) ;
      ++pmsg ;
   }
   if (rowMax) {
      /* semi-contiguous memory */
      int fromRank = myRank + domain.tp() ;
      int recvCount = dx * dz * xferFields ;
      MPI_Irecv(domain.commDataRecvView.data() + pmsg * maxPlaneComm,
                recvCount, baseType, fromRank, msgType,
                MPI_COMM_WORLD, &domain.recvRequest[pmsg]) ;
      ++pmsg ;
   }
   if (colMin && doRecv) {
      /* scattered memory */
      int fromRank = myRank - 1 ;
      int recvCount = dy * dz * xferFields ;
      MPI_Irecv(domain.commDataRecvView.data() + pmsg * maxPlaneComm,
                recvCount, baseType, fromRank, msgType,
                MPI_COMM_WORLD, &domain.recvRequest[pmsg]) ;
      ++pmsg ;
   }
   if (colMax) {
      /* scattered memory */
      int fromRank = myRank + 1 ;
      int recvCount = dy * dz * xferFields ;
      MPI_Irecv(domain.commDataRecvView.data() + pmsg * maxPlaneComm,
                recvCount, baseType, fromRank, msgType,
                MPI_COMM_WORLD, &domain.recvRequest[pmsg]) ;
      ++pmsg ;
   }

   if (!planeOnly) {
      /* receive data from domains connected only by an edge */
      if (rowMin && colMin && doRecv) {
         int fromRank = myRank - domain.tp() - 1 ;
         MPI_Irecv(domain.commDataRecvView.data() + pmsg * maxPlaneComm + emsg * maxEdgeComm,
                   dz * xferFields, baseType, fromRank, msgType,
                   MPI_COMM_WORLD, &domain.recvRequest[pmsg+emsg]) ;
         ++emsg ;
      }

      if (rowMin && planeMin && doRecv) {
         int fromRank = myRank - domain.tp()*domain.tp() - domain.tp() ;
         MPI_Irecv(domain.commDataRecvView.data() + pmsg * maxPlaneComm + emsg * maxEdgeComm,
                   dx * xferFields, baseType, fromRank, msgType,
                   MPI_COMM_WORLD, &domain.recvRequest[pmsg+emsg]) ;
         ++emsg ;
      }

      if (colMin && planeMin && doRecv) {
         int fromRank = myRank - domain.tp()*domain.tp() - 1 ;
         MPI_Irecv(domain.commDataRecvView.data() + pmsg * maxPlaneComm + emsg * maxEdgeComm,
                   dy * xferFields, baseType, fromRank, msgType,
                   MPI_COMM_WORLD, &domain.recvRequest[pmsg+emsg]) ;
         ++emsg ;
      }

      if (rowMax && colMax) {
         int fromRank = myRank + domain.tp() + 1 ;
         MPI_Irecv(domain.commDataRecvView.data() + pmsg * maxPlaneComm + emsg * maxEdgeComm,
                   dz * xferFields, baseType, fromRank, msgType,
                   MPI_COMM_WORLD, &domain.recvRequest[pmsg+emsg]) ;
         ++emsg ;
      }

      if (rowMax && planeMax) {
         int fromRank = myRank + domain.tp()*domain.tp() + domain.tp() ;
         MPI_Irecv(domain.commDataRecvView.data() + pmsg * maxPlaneComm + emsg * maxEdgeComm,
                   dx * xferFields, baseType, fromRank, msgType,
                   MPI_COMM_WORLD, &domain.recvRequest[pmsg+emsg]) ;
         ++emsg ;
      }

      if (colMax && planeMax) {
         int fromRank = myRank + domain.tp()*domain.tp() + 1 ;
         MPI_Irecv(domain.commDataRecvView.data() + pmsg * maxPlaneComm + emsg * maxEdgeComm,
                   dy * xferFields, baseType, fromRank, msgType,
                   MPI_COMM_WORLD, &domain.recvRequest[pmsg+emsg]) ;
         ++emsg ;
      }

      if (rowMax && colMin) {
         int fromRank = myRank + domain.tp() - 1 ;
          MPI_Irecv(domain.commDataRecvView.data() + pmsg * maxPlaneComm + emsg * maxEdgeComm,
                   dz * xferFields, baseType, fromRank, msgType,
                   MPI_COMM_WORLD, &domain.recvRequest[pmsg+emsg]) ;
         ++emsg ;
      }

      if (rowMin && planeMax) {
         int fromRank = myRank + domain.tp()*domain.tp() - domain.tp() ;
         MPI_Irecv(domain.commDataRecvView.data() + pmsg * maxPlaneComm + emsg * maxEdgeComm,
                   dx * xferFields, baseType, fromRank, msgType,
                   MPI_COMM_WORLD, &domain.recvRequest[pmsg+emsg]) ;
         ++emsg ;
      }

      if (colMin && planeMax) {
         int fromRank = myRank + domain.tp()*domain.tp() - 1 ;
         MPI_Irecv(domain.commDataRecvView.data() + pmsg * maxPlaneComm + emsg * maxEdgeComm,
                   dy * xferFields, baseType, fromRank, msgType,
                   MPI_COMM_WORLD, &domain.recvRequest[pmsg+emsg]) ;
         ++emsg ;
      }

      if (rowMin && colMax && doRecv) {
         int fromRank = myRank - domain.tp() + 1 ;
         MPI_Irecv(domain.commDataRecvView.data() + pmsg * maxPlaneComm + emsg * maxEdgeComm,
                   dz * xferFields, baseType, fromRank, msgType,
                   MPI_COMM_WORLD, &domain.recvRequest[pmsg+emsg]) ;
         ++emsg ;
      }

      if (rowMax && planeMin && doRecv) {
         int fromRank = myRank - domain.tp()*domain.tp() + domain.tp() ;
         MPI_Irecv(domain.commDataRecvView.data() + pmsg * maxPlaneComm + emsg * maxEdgeComm,
                   dx * xferFields, baseType, fromRank, msgType,
                   MPI_COMM_WORLD, &domain.recvRequest[pmsg+emsg]) ;
         ++emsg ;
      }

      if (colMax && planeMin && doRecv) {
         int fromRank = myRank - domain.tp()*domain.tp() + 1 ;
         MPI_Irecv(domain.commDataRecvView.data() + pmsg * maxPlaneComm + emsg * maxEdgeComm,
                   dy * xferFields, baseType, fromRank, msgType,
                   MPI_COMM_WORLD, &domain.recvRequest[pmsg+emsg]) ;
         ++emsg ;
      }

      /* receive data from domains connected only by a corner */
      if (rowMin && colMin && planeMin && doRecv) {
         /* corner at domain logical coord (0, 0, 0) */
         int fromRank = myRank - domain.tp()*domain.tp() - domain.tp() - 1 ;
         MPI_Irecv(domain.commDataRecvView.data() + pmsg * maxPlaneComm + emsg * maxEdgeComm + cmsg * CACHE_COHERENCE_PAD_REAL,
                   xferFields, baseType, fromRank, msgType,
                   MPI_COMM_WORLD, &domain.recvRequest[pmsg+emsg+cmsg]) ;
         ++cmsg ;
      }
      if (rowMin && colMin && planeMax) {
         /* corner at domain logical coord (0, 0, 1) */
         int fromRank = myRank + domain.tp()*domain.tp() - domain.tp() - 1 ;
         MPI_Irecv(domain.commDataRecvView.data() + pmsg * maxPlaneComm + emsg * maxEdgeComm + cmsg * CACHE_COHERENCE_PAD_REAL,
                   xferFields, baseType, fromRank, msgType,
                   MPI_COMM_WORLD, &domain.recvRequest[pmsg+emsg+cmsg]) ;
         ++cmsg ;
      }
      if (rowMin && colMax && planeMin && doRecv) {
         /* corner at domain logical coord (1, 0, 0) */
         int fromRank = myRank - domain.tp()*domain.tp() - domain.tp() + 1 ;
         MPI_Irecv(domain.commDataRecvView.data() + pmsg * maxPlaneComm + emsg * maxEdgeComm + cmsg * CACHE_COHERENCE_PAD_REAL,
                   xferFields, baseType, fromRank, msgType,
                   MPI_COMM_WORLD, &domain.recvRequest[pmsg+emsg+cmsg]) ;
         ++cmsg ;
      }
      if (rowMin && colMax && planeMax) {
         /* corner at domain logical coord (1, 0, 1) */
         int fromRank = myRank + domain.tp()*domain.tp() - domain.tp() + 1 ;
         MPI_Irecv(domain.commDataRecvView.data() + pmsg * maxPlaneComm + emsg * maxEdgeComm + cmsg * CACHE_COHERENCE_PAD_REAL,
                   xferFields, baseType, fromRank, msgType,
                   MPI_COMM_WORLD, &domain.recvRequest[pmsg+emsg+cmsg]) ;
         ++cmsg ;
      }
      if (rowMax && colMin && planeMin && doRecv) {
         /* corner at domain logical coord (0, 1, 0) */
         int fromRank = myRank - domain.tp()*domain.tp() + domain.tp() - 1 ;
         MPI_Irecv(domain.commDataRecvView.data() + pmsg * maxPlaneComm + emsg * maxEdgeComm + cmsg * CACHE_COHERENCE_PAD_REAL,
                   xferFields, baseType, fromRank, msgType,
                   MPI_COMM_WORLD, &domain.recvRequest[pmsg+emsg+cmsg]) ;
         ++cmsg ;
      }
      if (rowMax && colMin && planeMax) {
         /* corner at domain logical coord (0, 1, 1) */
         int fromRank = myRank + domain.tp()*domain.tp() + domain.tp() - 1 ;
         MPI_Irecv(domain.commDataRecvView.data() + pmsg * maxPlaneComm + emsg * maxEdgeComm + cmsg * CACHE_COHERENCE_PAD_REAL,
                   xferFields, baseType, fromRank, msgType,
                   MPI_COMM_WORLD, &domain.recvRequest[pmsg+emsg+cmsg]) ;
         ++cmsg ;
      }
      if (rowMax && colMax && planeMin && doRecv) {
         /* corner at domain logical coord (1, 1, 0) */
         int fromRank = myRank - domain.tp()*domain.tp() + domain.tp() + 1 ;
         MPI_Irecv(domain.commDataRecvView.data() + pmsg * maxPlaneComm + emsg * maxEdgeComm + cmsg * CACHE_COHERENCE_PAD_REAL,
                   xferFields, baseType, fromRank, msgType,
                   MPI_COMM_WORLD, &domain.recvRequest[pmsg+emsg+cmsg]) ;
         ++cmsg ;
      }
      if (rowMax && colMax && planeMax) {
         /* corner at domain logical coord (1, 1, 1) */
         int fromRank = myRank + domain.tp()*domain.tp() + domain.tp() + 1 ;
         MPI_Irecv(domain.commDataRecvView.data() + pmsg * maxPlaneComm + emsg * maxEdgeComm + cmsg * CACHE_COHERENCE_PAD_REAL,
                   xferFields, baseType, fromRank, msgType,
                   MPI_COMM_WORLD, &domain.recvRequest[pmsg+emsg+cmsg]) ;
         ++cmsg ;
      }
   }

   // printf("Posted CommRecv (%d plane, %d edge, %d corner messages)\n",
   //       pmsg, emsg, cmsg) ;
}

/******************************************/

void Copy1D(Kokkos::View<Real_t*> src, int src_offset,
   int src_stride,
    Kokkos::View<Real_t*> dest, int dst_offset, int dst_stride,
   int size, ExecSpace execSpace)
{
   Kokkos::parallel_for("Copy1D", RangePolicy(execSpace, 0, size),
                       KOKKOS_LAMBDA(const int i) {
      dest[dst_offset + i * dst_stride] = src[src_offset + i * src_stride];
   });
   Kokkos::fence();
}

void Add1D(Kokkos::View<Real_t*> src, int src_offset, int src_stride,
   Kokkos::View<Real_t*> dest, int dst_offset, int dst_stride,
   int size, ExecSpace execSpace)
{
   Kokkos::parallel_for("Add1D", RangePolicy(execSpace, 0, size),
                       KOKKOS_LAMBDA(const int i) {
      dest[dst_offset + i * dst_stride] += src[src_offset + i * src_stride];
   });
   Kokkos::fence();
}

/******************************************/

void Copy2D(Kokkos::View<Real_t*> src, 
   int src_offset,
   int src_stride_x, int src_stride_y,
   Kokkos::View<Real_t*> dest, 
   int dst_offset,
   int dst_stride_x, int dst_stride_y,
   int dim_x, int dim_y, ExecSpace execSpace)
{
   Kokkos::MDRangePolicy<Kokkos::Rank<2>> policy(execSpace, {0, 0}, {dim_x, dim_y});
   Kokkos::parallel_for("Copy2D", policy,
                       KOKKOS_LAMBDA(const int i, const int j) {
      dest[dst_offset + j * dst_stride_y + i * dst_stride_x] = 
         src[src_offset + j * src_stride_y + i * src_stride_x];
   });
}

void Add2D(Kokkos::View<Real_t*> src, 
   int src_offset,
   int src_stride_x, int src_stride_y,
   Kokkos::View<Real_t*> dest, 
   int dst_offset,
   int dst_stride_x, int dst_stride_y,
   int dim_x, int dim_y, ExecSpace execSpace)
{
   Kokkos::MDRangePolicy<Kokkos::Rank<2>> policy(execSpace, {0, 0}, {dim_x, dim_y});
   Kokkos::parallel_for("Add2D", policy,
                       KOKKOS_LAMBDA(const int i, const int j) {
      dest[dst_offset + j * dst_stride_y + i * dst_stride_x] += 
         src[src_offset + j * src_stride_y + i * src_stride_x];
   });
}

/******************************************/


void DomainChare::CommDataInit(Domain& domain, Index_t dx, Index_t dy, Index_t dz, 
                               bool doSend, bool planeOnly, CommDataMap_t &commDataMap)
{
   if (domain.numRanks() == 1)
      return ;

   /* post recieve buffers for all incoming messages */
   int myRank ;
   Index_t maxPlaneComm = domain.maxPlaneSize() ;
   Index_t maxEdgeComm  = domain.maxEdgeSize() ;
   Index_t pmsg = 0 ; /* plane comm msg */
   Index_t emsg = 0 ; /* edge comm msg */
   Index_t cmsg = 0 ; /* corner comm msg */
   //MPI_Datatype baseType = ((sizeof(Real_t) == 4) ? MPI_FLOAT : MPI_DOUBLE) ;
   bool rowMin, rowMax, colMin, colMax, planeMin, planeMax ;
   /* assume communication to 6 neighbors by default */
   rowMin = rowMax = colMin = colMax = planeMin = planeMax = true ;
   if (domain.rowLoc() == 0) {
      rowMin = false ;
   }
   if (domain.rowLoc() == (domain.tp()-1)) {
      rowMax = false ;
   }
   if (domain.colLoc() == 0) {
      colMin = false ;
   }
   if (domain.colLoc() == (domain.tp()-1)) {
      colMax = false ;
   }
   if (domain.planeLoc() == 0) {
      planeMin = false ;
   }
   if (domain.planeLoc() == (domain.tp()-1)) {
      planeMax = false ;
   }

   if (planeMin | planeMax) {
      /* ASSUMING ONE DOMAIN PER RANK, CONSTANT BLOCK SIZE HERE */
      int sendCount = dx * dy ;

      if (planeMin) {
         commDataMap[{thisIndex.x, thisIndex.y, thisIndex.z-1}] = CommData(
            pmsg, emsg, cmsg, 0, 1, 0, 1, 0, sendCount);
         ++pmsg ;
      }
      if (planeMax && doSend) {
         commDataMap[{thisIndex.x, thisIndex.y, thisIndex.z+1}] = CommData(
            pmsg, emsg, cmsg, dx*dy*(dz - 1), 1, 0, 1, 0, sendCount);
         ++pmsg ;
      }
   }
   if (rowMin | rowMax) {
      /* ASSUMING ONE DOMAIN PER RANK, CONSTANT BLOCK SIZE HERE */
      int sendCount = dx * dz ;

      if (rowMin) {
         commDataMap[{thisIndex.x, thisIndex.y-1, thisIndex.z}] = CommData(
            pmsg, emsg, cmsg, 0, 1, dx, 1, dx*dy, sendCount);
         ++pmsg ;
      }
      if (rowMax && doSend) {
         commDataMap[{thisIndex.x, thisIndex.y+1, thisIndex.z}] = CommData(
            pmsg, emsg, cmsg, dx*dy*(dy - 1), 1, dx, 1, dx*dy, sendCount);
         ++pmsg ;
      }
   }
   if (colMin | colMax) {
      /* ASSUMING ONE DOMAIN PER RANK, CONSTANT BLOCK SIZE HERE */
      int sendCount = dy * dz ;

      if (colMin) {
         commDataMap[{thisIndex.x-1, thisIndex.y, thisIndex.z}] = CommData(
            pmsg, emsg, cmsg, 0, 1, dy, dx, dx*dy, sendCount);
         ++pmsg ;
      }
      if (colMax && doSend) {
         commDataMap[{thisIndex.x+1, thisIndex.y, thisIndex.z}] = CommData(
            pmsg, emsg, cmsg, dx - 1, 1, dy, dx, dx*dy, sendCount);
         ++pmsg ;
      }
   }

   if (!planeOnly) {
      if (rowMin && colMin) {
         commDataMap[{thisIndex.x-1, thisIndex.y-1, thisIndex.z}] = 
            CommData(pmsg, emsg, cmsg, 0, 1, 0, 1, 0, dz);
         ++emsg ;
      }

      if (rowMin && planeMin) {
         commDataMap[{thisIndex.x, thisIndex.y-1, thisIndex.z-1}] = 
            CommData(pmsg, emsg, cmsg, 0, 1, 0, 1, 0, dx);
         ++emsg ;
      }

      if (colMin && planeMin) {
         commDataMap[{thisIndex.x-1, thisIndex.y, thisIndex.z-1}] = 
            CommData(pmsg, emsg, cmsg, 0, 1, 0, 1, 0, dy);
         ++emsg ;
      }

      if (rowMax && colMax && doSend) {
         commDataMap[{thisIndex.x+1, thisIndex.y+1, thisIndex.z}] = 
            CommData(pmsg, emsg, cmsg, dx*dy*(dz - 1), 1, 0, 1, 0, dz);
         ++emsg ;
      }

      if (rowMax && planeMax && doSend) {
         commDataMap[{thisIndex.x, thisIndex.y+1, thisIndex.z+1}] = 
            CommData(pmsg, emsg, cmsg, dx*dy*(dz - 1), 1, 0, 1, 0, dx);
         ++emsg ;
      }

      if (colMax && planeMax && doSend) {
         commDataMap[{thisIndex.x+1, thisIndex.y, thisIndex.z+1}] = 
            CommData(pmsg, emsg, cmsg, dx*dy*(dz - 1), 1, 0, 1, 0, dy);
         ++emsg ;
      }

      if (rowMax && colMin && doSend) {
         commDataMap[{thisIndex.x-1, thisIndex.y+1, thisIndex.z}] = 
            CommData(pmsg, emsg, cmsg, dx*dy*(dz - 1), 1, 0, 1, 0, dz);
         ++emsg ;
      }

      if (rowMin && planeMax && doSend) {
         commDataMap[{thisIndex.x, thisIndex.y-1, thisIndex.z+1}] = 
            CommData(pmsg, emsg, cmsg, dx*dy*(dz - 1), 1, 0, 1, 0, dx);
         ++emsg ;
      }

      if (colMin && planeMax && doSend) {
         commDataMap[{thisIndex.x-1, thisIndex.y, thisIndex.z+1}] = 
            CommData(pmsg, emsg, cmsg, dx*dy*(dz - 1), 1, 0, 1, 0, dy);
         ++emsg ;
      }

      if (rowMin && colMax) {
         commDataMap[{thisIndex.x+1, thisIndex.y-1, thisIndex.z}] = 
            CommData(pmsg, emsg, cmsg, dx*dy*(dz - 1), 1, 0, 1, 0, dz);
         ++emsg ;
      }

      if (rowMax && planeMin) {
         commDataMap[{thisIndex.x, thisIndex.y+1, thisIndex.z-1}] = 
            CommData(pmsg, emsg, cmsg, dx*(dy - 1), 1, 0, 1, 0, dx);
         ++emsg ;
      }

      if (colMax && planeMin) {
         commDataMap[{thisIndex.x+1, thisIndex.y, thisIndex.z-1}] = 
            CommData(pmsg, emsg, cmsg, dx - 1, 1, 0, 1, 0, dy);
         ++emsg ;
      }

      if (rowMin && colMin && planeMin) {
         /* corner at domain logical coord (0, 0, 0) */
         commDataMap[{thisIndex.x-1, thisIndex.y-1, thisIndex.z-1}] = 
            CommData(pmsg, emsg, cmsg, 0, 1, 0, 1, 0, 1);
         ++cmsg ;
      }
      if (rowMin && colMin && planeMax && doSend) {
         /* corner at domain logical coord (0, 0, 1) */
         Index_t idx = dx*dy*(dz - 1) ;
         commDataMap[{thisIndex.x-1, thisIndex.y-1, thisIndex.z+1}] = 
            CommData(pmsg, emsg, cmsg, idx, 1, 0, 1, 0, 1);
         ++cmsg ;
      }
      if (rowMin && colMax && planeMin) {
         /* corner at domain logical coord (1, 0, 0) */
         Index_t idx = dx - 1 ;
         commDataMap[{thisIndex.x+1, thisIndex.y-1, thisIndex.z-1}] = 
            CommData(pmsg, emsg, cmsg, idx, 1, 0, 1, 0, 1);
         ++cmsg ;
      }
      if (rowMin && colMax && planeMax && doSend) {
         /* corner at domain logical coord (1, 0, 1) */
         Index_t idx = dx*dy*(dz - 1) + (dx - 1) ;
         commDataMap[{thisIndex.x+1, thisIndex.y-1, thisIndex.z+1}] = 
            CommData(pmsg, emsg, cmsg, idx, 1, 0, 1, 0, 1);
         ++cmsg ;
      }
      if (rowMax && colMin && planeMin) {
         /* corner at domain logical coord (0, 1, 0) */
         Index_t idx = dx*(dy - 1) ;
         commDataMap[{thisIndex.x-1, thisIndex.y+1, thisIndex.z-1}] = 
            CommData(pmsg, emsg, cmsg, idx, 1, 0, 1, 0, 1);
         ++cmsg ;
      }
      if (rowMax && colMin && planeMax && doSend) {
         /* corner at domain logical coord (0, 1, 1) */
         Index_t idx = dx*dy*(dz - 1) + dx*(dy - 1) ;
         commDataMap[{thisIndex.x-1, thisIndex.y+1, thisIndex.z+1}] = 
            CommData(pmsg, emsg, cmsg, idx, 1, 0, 1, 0, 1);
         ++cmsg ;
      }
      if (rowMax && colMax && planeMin) {
         /* corner at domain logical coord (1, 1, 0) */
         Index_t idx = dx*dy - 1 ;
         commDataMap[{thisIndex.x+1, thisIndex.y+1, thisIndex.z-1}] = 
            CommData(pmsg, emsg, cmsg, idx, 1, 0, 1, 0, 1);
         ++cmsg ;
      }
      if (rowMax && colMax && planeMax && doSend) {
         /* corner at domain logical coord (1, 1, 1) */
         Index_t idx = dx*dy*dz - 1 ;
         commDataMap[{thisIndex.x+1, thisIndex.y+1, thisIndex.z+1}] = 
            CommData(pmsg, emsg, cmsg, idx, 1, 0, 1, 0, 1);
         ++cmsg ;
      }
   }
}

void DomainChare::CommSend(Domain& domain, int msgType,
                           Index_t xferFields, Kokkos::View<Real_t*> *fieldData,
                           Index_t dx, Index_t dy, Index_t dz, bool doSend, bool planeOnly,
                           CommDataMap_t& commDataMap)
{
   if (domain.numRanks() == 1)
      return ;

   Index_t maxPlaneComm = xferFields * domain.maxPlaneSize() ;
   Index_t maxEdgeComm  = xferFields * domain.maxEdgeSize() ;

   CommDataMapIter_t it;

   for (it = commDataMap.begin(); it != commDataMap.end(); ++it) {
      std::tuple<int, int, int> idx = it->first ;
      CommData cdata = it->second ;
      int offsetX = std::get<0>(idx) - thisIndex.x ;
      int offsetY = std::get<1>(idx) - thisIndex.y ;
      int offsetZ = std::get<2>(idx) - thisIndex.z ;

      if (((offsetX == -1 || offsetX == 1) && offsetY == 0 && offsetZ == 0) || 
         offsetX == 0 && ((offsetY == -1 || offsetY == 1) && offsetZ == 0)) {
         for (Index_t fi=0 ; fi<xferFields; ++fi) {
            Kokkos::View<Real_t*> src = fieldData[fi] ;
            Copy2D(src, cdata.offset, cdata.dst_stride[0], cdata.dst_stride[1],
                   domain.commDataSendView, 
                   cdata.pmsg * maxPlaneComm + cdata.emsg * maxEdgeComm + cdata.cmsg * CACHE_COHERENCE_PAD_REAL + fi * cdata.size, 
                   cdata.src_stride[0], cdata.src_stride[1], cdata.size);
         }
      } else {
         for (Index_t fi=0 ; fi<xferFields; ++fi) {
            Kokkos::View<Real_t*> src = fieldData[fi] ;
            Copy1D(src, cdata.offset, cdata.dst_stride[0],
                   domain.commDataSendView, 
                   cdata.pmsg * maxPlaneComm + cdata.emsg * maxEdgeComm + cdata.cmsg * CACHE_COHERENCE_PAD_REAL + fi * cdata.size, 
                   cdata.src_stride[0], cdata.size);
         }
      }
   
      CkCallback* cb = new CkCallback(
         packingDoneCallback, 
         new PackingDoneMsg(thisProxy, msgType, std::get<0>(idx), std::get<1>(idx), 
            std::get<2>(idx), xferFields, cdata.sendCount, cdata.sendOffset)
      );
      hapiAddCallback(commStream, cb);
   }
}

extern "C" void packingDoneCallback(void* param, void* msg) {
   PackingDoneMsg* m = (PackingDoneMsg*) msg;
   CProxy_DomainChare* proxy = m->proxy;
   proxy->packingDone(m->msgType, m->x, m->y, m->z, m->xferFields, m->sendCount, m->offset);
}

void DomainChare::packingDone(int msgType, int x, int y, int z, int xferFields, 
      int sendCount, int offset) {
   int ref = msgType | iter;
   thisProxy(x, y, z).CommRecv(ref, thisIndex.x, thisIndex.y, thisIndex.z, 
      xferFields, sendCount, domain.commDataSendView.data() + offset);
}

/******************************************/

void DomainChare::CommRecv(int& ref, int& x, int& y, int& z, int& xferFields, int& size, Real_t* &buf, CkDeviceBufferPost* post) {
   int msgType = ref >> 29;
   CommDataMap_t* commDataMap;
   if (msgType == MSG_COMM_POSVEL)
      commDataMap = &commDataPosVel;
   else if (msgType == MSG_COMM_Q)
      commDataMap = &commDataQ;
   else if (msgType == MSG_COMM_SBN)
      commDataMap = &commDataSBN;
   else
      CkAbort("DomainChare::CommRecv: Unknown msgType") ;

   CommDataMapIter_t it = commDataMap->find({x, y, z});
   if (it == commDataMap->end())
      CkAbort("DomainChare::CommRecv: Invalid comm data map key") ;

   Index_t maxPlaneComm = xferFields * locDom->maxPlaneSize() ;
   Index_t maxEdgeComm  = xferFields * locDom->maxEdgeSize() ;

   int pmsg = it->second.pmsg ;
   int emsg = it->second.emsg ;
   int cmsg = it->second.cmsg ;

   buf = locDom->commDataRecvView.data() + pmsg * maxPlaneComm + emsg * maxEdgeComm + cmsg * CACHE_COHERENCE_PAD_REAL;
   post[0].cuda_stream = commStream;
}

/******************************************/

void DomainChare::processRemotePosVel(int ref, int x, int y, int z, int xferFields, int size, Real_t* buf) {
   Domain& domain = *locDom;

   Index_t maxPlaneComm = xferFields * domain.maxPlaneSize() ;
   Index_t maxEdgeComm  = xferFields * domain.maxEdgeSize() ;

   CommData& cdata = commDataPosVel[{x, y, z}];
   Index_t offsetX = x - thisIndex.x;
   Index_t offsetY = y - thisIndex.y;
   Index_t offsetZ = z - thisIndex.z;

   Kokkos::View<Real_t*> fieldData[6];
   fieldData[0] = domain.m_x ;
   fieldData[1] = domain.m_y ;
   fieldData[2] = domain.m_z ;
   fieldData[3] = domain.m_xd ;
   fieldData[4] = domain.m_yd ;
   fieldData[5] = domain.m_zd ;
   //int xferFields = 6 ;

   if (((offsetX == -1 || offsetX == 1) && offsetY == 0 && offsetZ == 0) || 
         offsetX == 0 && ((offsetY == -1 || offsetY == 1) && offsetZ == 0)) {
      for (Index_t fi=0 ; fi<xferFields; ++fi) {
         Kokkos::View<Real_t*> dest = fieldData[fi] ;
         Copy2D(dest,
            commData.dst_offset, commData.dst_stride[0], commData.dst_stride[1],
            domain.commDataRecvView, commData.src_offset + fi * commData.size, 
            commData.src_stride[0], commData.src_stride[1],
            commData.size);
      }
   } else {
      for (Index_t fi=0 ; fi<xferFields; ++fi) {
         Kokkos::View<Real_t*> dest = fieldData[fi] ;
         Copy1D(domain.commDataRecvView, 
            cdata.pmsg * maxPlaneComm + cdata.emsg * maxEdgeComm + cdata.cmsg * CACHE_COHERENCE_PAD_REAL + fi * cdata.size,
            cdata.src_stride[0],
            dest, cdata.offset, cdata.dst_stride[0],
            cdata.size
            );
      }
   }
}

/******************************************/

void DomainChare::processRemoteQ(int ref, int x, int y, int z, int xferFields, int size, Real_t* buf) {
   Domain& domain = *locDom;

   CommData commData = commDataQ[{x, y, z}];
   Index_t offsetX = x - thisIndex.x;
   Index_t offsetY = y - thisIndex.y;
   Index_t offsetZ = z - thisIndex.z;

   Kokkos::View<Real_t*> fieldData[3];
   Index_t fieldOffset[3];
   fieldData[0] = domain.m_delv_xi ;
   fieldData[1] = domain.m_delv_eta ;
   fieldData[2] = domain.m_delv_zeta ;
   fieldOffset[0] = domain.numElem() ;
   fieldOffset[1] = domain.numElem() ;
   fieldOffset[2] = domain.numElem() ;
   int xferFields = 3;

   if (((offsetX == -1 || offsetX == 1) && offsetY == 0 && offsetZ == 0) || 
         offsetX == 0 && ((offsetY == -1 || offsetY == 1) && offsetZ == 0)) {
      for (Index_t fi=0 ; fi<xferFields; ++fi) {
         Kokkos::View<Real_t*> dest = fieldData[fi] ;
         Copy2D(dest,
            commData.dst_offset, commData.dst_stride[0], commData.dst_stride[1],
            domain.commDataRecvView, commData.src_offset + fi * commData.size, 
            commData.src_stride[0], commData.src_stride[1],
            commData.size);
      }
   } else {
      for (Index_t fi=0 ; fi<xferFields; ++fi) {
         Kokkos::View<Real_t*> dest = fieldData[fi] ;
         Copy1D(dest,
            commData.dst_offset, commData.dst_stride[0],
            domain.commDataRecvView, commData.src_offset + fi * commData.size, 
            commData.src_stride[0],
            commData.size);
      }
   }
}

/******************************************/

void DomainChare::processRemoteMass(int ref, int x, int y, int z, int xferFields, int size, Real_t* buf) {
   Domain& domain = *locDom;

   CommData cdata = commDataSBN[{x, y, z}];
   Index_t offsetX = x - thisIndex.x;
   Index_t offsetY = y - thisIndex.y;
   Index_t offsetZ = z - thisIndex.z;

   Kokkos::View<Real_t*> fieldData[1];
   fieldData[0] = domain.m_nodalMass;

   if (((offsetX == -1 || offsetX == 1) && offsetY == 0 && offsetZ == 0) || 
         offsetX == 0 && ((offsetY == -1 || offsetY == 1) && offsetZ == 0)) {
      for (Index_t fi=0 ; fi<xferFields; ++fi) {
         Kokkos::View<Real_t*> dest = fieldData[fi] ;
         Add2D(dest, cdata.dst_offset, cdata.dst_stride[0], cdata.dst_stride[1],
               domain.commDataRecvView, cdata.src_offset + fi * cdata.size, 
               cdata.src_stride[0], cdata.src_stride[1],
               cdata.size);
      }
   } else {
      for (Index_t fi=0 ; fi<xferFields; ++fi) {
         Kokkos::View<Real_t*> dest = fieldData[fi] ;
         Add1D(dest, cdata.dst_offset, cdata.dst_stride[0],
               domain.commDataRecvView, cdata.src_offset + fi * cdata.size, 
               cdata.src_stride[0],
               cdata.size);
      }
   }
}

/******************************************/

void DomainChare::processRemoteForce(int ref, int x, int y, int z, int xferFields, int size, Real_t* buf) {
   Domain& domain = *locDom;

   CommData cdata = commDataSBN[{x, y, z}];
   Index_t offsetX = x - thisIndex.x;
   Index_t offsetY = y - thisIndex.y;
   Index_t offsetZ = z - thisIndex.z;

   Kokkos::View<Real_t*> fieldData[3];
   fieldData[0] = domain.m_fx;
   fieldData[1] = domain.m_fy;
   fieldData[2] = domain.m_fz;

   if (((offsetX == -1 || offsetX == 1) && offsetY == 0 && offsetZ == 0) || 
         offsetX == 0 && ((offsetY == -1 || offsetY == 1) && offsetZ == 0)) {
      for (Index_t fi=0 ; fi<xferFields; ++fi) {
         Kokkos::View<Real_t*> dest = fieldData[fi] ;
         Add2D(dest, cdata.dst_offset, cdata.dst_stride[0], cdata.dst_stride[1],
               domain.commDataRecvView, cdata.src_offset + fi * cdata.size, 
               cdata.src_stride[0], cdata.src_stride[1],
               cdata.size);
      }
   } else {
      for (Index_t fi=0 ; fi<xferFields; ++fi) {
         Kokkos::View<Real_t*> dest = fieldData[fi] ;
         Add1D(dest, cdata.dst_offset, cdata.dst_stride[0],
               domain.commDataRecvView, cdata.src_offset + fi * cdata.size, 
               cdata.src_stride[0],
               cdata.size);
      }
   }
}
