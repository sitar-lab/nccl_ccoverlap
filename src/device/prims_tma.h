/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "common.h"
#include "network/unpack/unpack.h"
#include <cassert>
#include <cuda/barrier>
#include <cuda/pipeline>
#include <cuda/ptx>
#include <new>

// [jihwan] TMA Pipeline Implementation
// This file implements the TMA (Tensor Memory Accelerator) protocol for NCCL.
// All implementations are based on Simple protocol (prims_simple.h)
// Key features:
// 1. Pipelined Execution: Overlaps data transfer (TMA Load) with data transfer to peer GPU + computation (ReduceCopy).
// 2. Prologue/MainLoop/Epilogue Structure: Prefetches data in prologue, consumes in main loop.
// 3. Shared Memory Buffering: Loads data directly from global memory to SMEM using TMA.
// 4. Step Synchronization: Uses waitPeerForTmaLoad to handle out-of-order slice completion.


// [jihwan] TMA Debug output
#define NCCL_TMA_DEBUG 0

#if NCCL_TMA_DEBUG
  #define TMA_DEBUG_PRINT(fmt, ...) \
    if (tid == 0 && blockIdx.x == 0) { \
      printf("[TMA BLK=%d TID=%d] " fmt "\n", (int)blockIdx.x, (int)threadIdx.x, ##__VA_ARGS__); \
    }
#else
  #define TMA_DEBUG_PRINT(fmt, ...) 
#endif

static constexpr int NCCL_TMA_MAX_SMEM_BYTES = 100 * 1024;  // 100KB
enum primsTmaMode {
  primsTmaModeDefault = 0,
  primsTmaModePatRs = 1,
  primsTmaModePatAg = 2
};

template<typename T, typename RedOp, typename Fan, int Direct,
         int SlicePerChunk, int StepPerSlice, int Unroll, int P2p, int MultimemSrcs, int MultimemDsts, bool isNetOffload>
class Primitives<
    T, RedOp, Fan, Direct, ProtoTMA<SlicePerChunk, StepPerSlice, Unroll, MultimemSrcs, MultimemDsts>, P2p, isNetOffload
  > {
  static constexpr int MaxRecv = Fan::MaxRecv, MaxSend = Fan::MaxSend;
  static constexpr int Input=0, Output=1;
  static constexpr int RoleInput = 0x01,
                       RoleOutput = 0x02,
                       RoleWaitRecv = 0x04,
                       RoleWaitSend = 0x08,
                       RolePostSend = 0x10,
                       RolePostRecv = 0x20,
                       Aborted = 0x40,
                       NetRegMode = 0x80,
                       ConnFifoEnabled = 0x100,
                       DirectWrite = 0x200,
                       DirectRead = 0x400,
                       PatMode = 0x800,
                       NvlsMinPolling = 0x1000,
                       NetDeviceUnpack = 0x2000,
                       AnyNetDeviceUnpack = 0x4000,
                       RoleTmaLoad = 0x8000;  // TMA prefetch role
  const int tid, tidInBlock;
  const int nthreads;
  int nworkers;
  const int stepSize;
  Fan fan;
  int index; // Peer index I'm responsible for
  static constexpr int PipeDepth = NCCL_TMA_PIPE_DEPTH;
  int flags;
  int group;
  uint64_t step;
  struct ncclConnInfo* conn = NULL; // pointer to connection info
  struct ncclConnFifo* connFifo = NULL;
  T* connEltsFifo; // Address of peer FIFO buffer
  T* directBuff = NULL;
  uint64_t *connStepPtr;
  uint64_t connStepCache; // Cache last seen value of (*connStepPtr)
  int      connStepSize; // Connection step size
  void*    netDeviceHandle;
  uint64_t accSize;

  // Don't use barrier 0 as it's used by the final sync
  // [jihwan] barrier for all threads
  __device__ void barrier() {
    if (nthreads == WARP_SIZE) __syncwarp();
    else {
      int bar = 15-group;
      barrier_sync(bar, nthreads);
    }
  }

  // [jihwan] barrier only for worker threads
  __device__ void subBarrier() {
    if (nworkers == WARP_SIZE) __syncwarp();
    else {
      int bar = 15-group - (nworkers!=nthreads ? 1 : 0);
      barrier_sync(bar, nworkers);
    }
  }

  // PAT uses a single barrier across all groups
  __device__ void patBarrier() {
    barrier_sync(15, NCCL_PAT_NWORKERS);
  }

  __device__ bool barrierAny(int vote) {
    if (nthreads == WARP_SIZE) {
      return __any_sync(~0u, vote);
    } else {
      int name = 15-group;
      return barrier_red_or(vote, name, nthreads);
    }
  }
  __device__ bool subBarrierAny(int vote) {
    if (nworkers == WARP_SIZE) {
      return __any_sync(~0u, vote);
    } else {
      int name = 15-group - (nworkers!=nthreads ? 1 : 0);
      return barrier_red_or(vote, name, nworkers);
    }
  }

  inline __device__ uint64_t loadStepValue(uint64_t* ptr) {
    #if __CUDA_ARCH__ >= 900 && CUDART_VERSION >= 12010
    if (flags & NvlsMinPolling) {
      uint64_t ans;
      asm volatile("multimem.ld_reduce.acquire.sys.global.min.u64 %0, [%1];" : "=l"(ans) : "l"(cvta_to_global(ptr)) : "memory");
      return ans;
    }
    #endif
    // volatile is faster than acquire but not as correct. Make sure reduceCopy
    // loads data using volatile so it doesn't see stale data in L1.
    return ld_volatile_global(ptr);
  }

  /*
    offset : offset += sliceSize in each slice iteration
  */
  
  // Separate function for TMA prologue/produce: wait for specific slice's data
  template <int DirectRecv, int DirectSend, int Recv, int Send, int Src, int Dst>
  __device__ __forceinline__ void waitPeerForTmaLoad(intptr_t srcIx, intptr_t dstIx, int offset, int nelts, int sliceOffset) {
    
    // For Send:
    // This function is for "TMA Load" preparation (checking if source data is ready).
    // 1. If Src is UserInput (CopySend): Data is always assumed ready.
    // 2. If Src is FIFO (RecvSend): Data availability is checked in the Recv block above.
    // 3. Pointer setup for Src is done by the caller (tid=0 context in genericOp).
    // 4. Checking peer's tail (space availability) should be done in Main Loop, not here.
    // Therefore, Send role does NOTHING here to avoid pipeline stalls.

    // Early Exit for Non-Recv Roles
    if (!(flags & (Recv * RoleWaitRecv))) {
        return;
    }

    // [Fix] Calculate stride per slice dynamically based on roles (Wait+Post)
    int stepIncrement = 0;
    if (flags & (Recv*RoleWaitRecv | Send*RoleWaitSend)) stepIncrement += StepPerSlice;
    // if (flags & (Recv*RolePostRecv | Send*RolePostSend)) stepIncrement += StepPerSlice;

    // [Fix] sliceOffset is 0-based index (0,1,2..) for the current chunk.
    // Calculate absolute step correctly using StepPerSlice.
    uint64_t absStep = step + sliceOffset * stepIncrement;
    int buffSlot = absStep % NCCL_STEPS;
    
    #ifdef NCCL_TMA_DEBUG
    // if (threadIdx.x == 0 && sliceOffset < 4) {
    //      printf("TMA_DBG: R%d S%d A%llu I%d C%llu\n", ncclShmem.comm.rank, sliceOffset, absStep, stepIncrement, connStepCache);
    // }
    #endif
    
    // For Recv: Wait until peer has produced data for the specified slice step
    int spins = 0;
    while (connStepCache < absStep + StepPerSlice) { 
      connStepCache = loadStepValue(connStepPtr); // head
      if (checkAbort(flags, Aborted, spins)) break;
    }
    
    // Set up source pointers to peer's FIFO buffer at the specified step
    void **ptrs = ncclShmem.groups[group].srcs + Src;
    const int index = 0;

    if ((flags & ConnFifoEnabled) && connFifo[buffSlot].mode == NCCL_MODE_OFFSET) {
      ptrs[index] = connEltsFifo + loadInt(&connFifo[buffSlot].offset)/sizeof(T);
    } else if (DirectRecv) {
      if (flags & DirectRead) {
        ptrs[index] = directBuff + srcIx + offset;
      } else {
        ptrs[index] = connEltsFifo + buffSlot*connStepSize;
      }
    } else {
      ptrs[index] = connEltsFifo + buffSlot*connStepSize;
    }

  }
  
  // [jihwan]
  // Wait for peer to be ready for next slice (in GMEM view)
  // After executing this function, step is increased by StepPerSlice (only for wait roles threads)
  template <int DirectRecv, int DirectSend, int Recv, int Send, int Src, int Dst>
  __device__ __forceinline__ void waitPeer(intptr_t srcIx, intptr_t dstIx, int offset, int nelts) {
    
    /* [jihwan]
      Send:
        send = 1, recv = 0 : isSendNotRecv = 1
      Recv:
        send = 0, recv = 1 : isSendNotRecv = 0
      RecvSend:
        send = 1, recv = 1 : isSendNotRecv = flags & RoleWaitSend ? 1 : 0
        in recvsend case according to their tid some threads will wait for send and some for recv
    */
    const bool isSendNotRecv = (Send && Recv) ? (flags & RoleWaitSend) : Send;
    
    // [jihwan] Check if we are in TMA mode (Send Only) or Simple mode (Recv/RecvSend)
    // Send && !Recv && Src is the condition used in genericOp to enable TMA
    const bool isTmaMode = (Send && !Recv && Src);

    // Wait until peer has produced/consumed data
    // If waiting to recv, wait until peer has produced data for this slice (check head pointer)
    // If waiting to send, wait until peer has consumed data (check tail pointer)
    
    // [Optimization]
    // 1. Send Role: Must check Peer's Tail (Space Availability)
    // 2. Recv Role: Skip Head Check ONLY IF TMA IS USED. 
    //    If !isTmaMode, we must perform head check like standard Simple protocol.
    
    if (flags & (Send * RoleWaitSend)) {
      int spins = 0;
      while (connStepCache + NCCL_STEPS < step + StepPerSlice) {
        connStepCache = loadStepValue(connStepPtr);
        if (checkAbort(flags, Aborted, spins)) break;
      }
    } else if (flags & (Recv * RoleWaitRecv) && !isTmaMode) {
      // [jihwan] Restore Head Check for non-TMA path (Recv/RecvSend)
      int spins = 0;
      while (connStepCache < step + StepPerSlice) {
        connStepCache = loadStepValue(connStepPtr);
        if (checkAbort(flags, Aborted, spins)) break;
      }
    }

    // Guaranteed to use FIFO, set up ptrs now
    if (flags & (Recv*RoleWaitRecv | Send*RoleWaitSend)) {
      if ((flags & ConnFifoEnabled) && (flags & (Send * RoleWaitSend)))
        connFifo[step%NCCL_STEPS].size = nelts*sizeof(T);


      /* [jihwan]
        ptrs : array of pointers to srcs or dsts, store address of data to be sent or received in this slice
        if send func, we have to fill dest ptrs -> ptrs = dsts pointer
        if recv func, we have to fill source ptrs -> ptrs = srcs pointer

        Why we add Dst and Src to ptrs?
        - The first space for dsts and srcs is reserved for User buffer when it exists

        P2p : Activated when ncclSend or ncclRecv only
      */

      // [Optimization] Recv Role skips pointer setup here ONLY IF TMA IS USED.
      // If !isTmaMode, we must setup pointers.
      bool skipPtrSetup = !isSendNotRecv && isTmaMode;
      
      if (!skipPtrSetup) {
        void **ptrs = isSendNotRecv ? ncclShmem.groups[group].dsts + Dst
                                    : ncclShmem.groups[group].srcs + Src;
        
        /* [jihwan]
          Determine the actual pointer to use for this slice
          NetRegMode: Using registered memory for network transfer
        */
        TMA_DEBUG_PRINT("[waitPeer Func] NetRegMode: %d",flags & NetRegMode);
        TMA_DEBUG_PRINT("[waitPeer Func] ConnFifoEnabled: %d",flags & ConnFifoEnabled);

        if ((flags & NetRegMode) && DirectSend) {
          if (P2p) {
            ptrs[index] = NULL;
          } else {
             if (!Recv)
               ptrs[index] = NULL;
             else
               ptrs[index] = (T*)ncclShmem.groups[group].userOutput + dstIx + offset;
          }
        } else if ((flags & ConnFifoEnabled) && connFifo[step%NCCL_STEPS].mode == NCCL_MODE_OFFSET) {
          ptrs[index] = connEltsFifo + loadInt(&connFifo[step%NCCL_STEPS].offset)/sizeof(T);
        } else if (DirectSend && isSendNotRecv) { // DirectSend logic applies to Send role
          if (flags & DirectWrite) {
            ptrs[index] = directBuff + dstIx + offset;
          } else if (flags & DirectRead) {  // empty send
            ptrs[index] = nullptr;
          } else {
            ptrs[index] = connEltsFifo + (step%NCCL_STEPS)*connStepSize;
          }
        } else if (DirectRecv && !isSendNotRecv) { // DirectRecv logic applies to Recv role
           if (flags & DirectRead) {
             ptrs[index] = directBuff + srcIx + offset;
           } else {
             ptrs[index] = connEltsFifo + (step%NCCL_STEPS)*connStepSize;
           }
        }
        else {
          // Yes, for some template arguments this code will be unreachable.  That's fine.
          // coverity[dead_error_line]
          TMA_DEBUG_PRINT("[waitPeer Func] Reach that we intended");
          ptrs[index] = connEltsFifo + (step%NCCL_STEPS)*connStepSize;
        }
      } // End of Pointer Setup Logic

      if (flags & NetDeviceUnpack) {
        ncclNetDeviceIncrementHead(group, index);
      }
      #ifdef NCCL_TMA_DEBUG
      // if (threadIdx.x == 0 && step < 20) printf("WaitPeer: Step %llu -> %llu\n", step, step + StepPerSlice);
      #endif
      step += StepPerSlice;
    }
  }


  // [jihwan]
  // Wait for peer to complete data store for the slice
  // After executing this function, step is increased by StepPerSlice (only for post roles threads)
  template<int Recv, int Send>
  inline __device__ void postPeer(bool dataStored) {
    if (flags & (Recv*RolePostRecv | Send*RolePostSend)) {
      step += StepPerSlice;
      if (Send && (flags & RolePostSend) && (dataStored||(flags&ConnFifoEnabled))) {
        fence_acq_rel_sys();
      }
      st_relaxed_sys_global(connStepPtr, step);
    }
  }

  template <int DirectRecv1, int DirectSend1, int Recv, int Send, int SrcBuf, int DstBuf>
  __device__ __forceinline__ void genericOp(
      intptr_t srcIx, intptr_t dstIx, int nelem, bool postOp
    ) {
    constexpr int DirectRecv = 1 && Direct && DirectRecv1;
    constexpr int DirectSend = 1 && Direct && DirectSend1;
    /*
      Src: Using User buffer as src buffer
      Dst: Using User buffer as dst buffer
    */
    constexpr int Src = SrcBuf != -1;
    constexpr int Dst = DstBuf != -1;

    nelem = nelem < 0 ? 0 : nelem;
    int sliceSize = stepSize*StepPerSlice;
    sliceSize = max(divUp(nelem, 16*SlicePerChunk)*16, sliceSize/32);
    

    const int totalSlices = (nelem + sliceSize - 1) / sliceSize;
    
    TMA_DEBUG_PRINT("GENERICOP INIT: Src=%d, Dst=%d, SrcBuf=%d, DstBuf=%d", Src, Dst, SrcBuf, DstBuf);
    TMA_DEBUG_PRINT("GENERICOP INIT: nelem = %d, Slice Size=%d, SlicePerChunk=%d",nelem, sliceSize, SlicePerChunk);

    /* [jihwan] TMA Pipelined Implementation */
    // Sync with threads within the same block
    using tma_barrier_t = cuda::barrier<cuda::thread_scope_block>;
    
    // [jihwan]
    // Barrier storage using aligned_storage to avoid dynamic initialization in __device__ function
    // In this code, compiler allocate memory space for NCCL_TMA_PIPE_DEPTH barriers in __shared__ memory (Each thread don't execute constructor)
    __shared__ alignas(alignof(tma_barrier_t)) 
      std::aligned_storage_t<sizeof(tma_barrier_t), alignof(tma_barrier_t)> 
      barrierStorage[NCCL_TMA_PIPE_DEPTH];
    
    // [jihwan]
    // Each thread has its own token array for TMA barriers
    typename tma_barrier_t::arrival_token tmaTokens[NCCL_TMA_PIPE_DEPTH];
    
    // Determine if TMA should be used
    bool shouldUseTma = false;
    // [jihwan] Use TMA only for pure Send operations.
    // Recv and RecvSend fallback to simple path (no TMA) to avoid waiting for peer data which causes stalls.
    if (Send && !Recv && Src) {
      shouldUseTma = true;
    }

    if (tid < nworkers && totalSlices > 0 && !isNetOffload && shouldUseTma) {
      // ========================================
      // TMA PIPELINED PATH
      // ========================================
      
      // Initialize TMA barriers using placement new
      if (tid == 0) {
        #pragma unroll
        for (int i = 0; i < NCCL_TMA_PIPE_DEPTH; ++i) {
          new (&barrierStorage[i]) tma_barrier_t(nworkers);
        }
        #if __CUDA_ARCH__ >= 900
        cuda::ptx::fence_proxy_async(cuda::ptx::space_shared); // Ensure TMA barrier initialization is visible to async proxy(TMA path)
        #endif
        TMA_DEBUG_PRINT("TMA INIT: Initialized %d barriers, nworkers=%d", NCCL_TMA_PIPE_DEPTH, nworkers);
      }
      subBarrier(); // Wait for barrier initialization
      
      // ========================================
      // PROLOGUE: Preload first NCCL_TMA_PIPE_DEPTH slices
      // ========================================
      int offset = 0;
      int preloadCount = min(NCCL_TMA_PIPE_DEPTH, totalSlices);

      TMA_DEBUG_PRINT("TMA PROLOGUE: Initialized preloadCount=%d, totalSlices=%d", preloadCount, totalSlices);
      
      
      for (int preloadIdx = 0; preloadIdx < preloadCount; ++preloadIdx) {
        int currentSliceSize = sliceSize < nelem-offset ? sliceSize : nelem - offset;
        int tmaSlot = preloadIdx % NCCL_TMA_PIPE_DEPTH;
        
        // For Send operations, set source pointer to user buffer first
        if (tid == 0 && Send && Src) {
          T* userInput = (T*)ncclShmem.groups[group].userInput;
          T* userOutput = (T*)ncclShmem.groups[group].userOutput;
          ncclShmem.groups[group].srcs[0] = (SrcBuf==Input ? userInput : userOutput) + srcIx + offset;
        }
        
        // [TMA PROLOGUE] Wait for peer + setup pointers for slice index preloadIdx
        // Pass preloadIdx as the sliceStep so each slice waits for its own step
        waitPeerForTmaLoad<DirectRecv, DirectSend, Recv, Send, Src, Dst>(
          srcIx, dstIx, offset, currentSliceSize, /*sliceStep=*/preloadIdx);
        
        if (tid == 0) {
          TMA_DEBUG_PRINT("TMA PROLOGUE: preloadIdx=%d step=%ld srcPtr=%p connStepCache=%ld", 
                          preloadIdx, step, ncclShmem.groups[group].srcs[0], connStepCache);
        }
        
        tma_barrier_t* tmaBar = reinterpret_cast<tma_barrier_t*>(&barrierStorage[tmaSlot]);

        // Issue TMA load (thread 0 only)
        if (tid == 0) {
          void* tmaShmemSlot = ncclTmaShmemSlot(tmaSlot, NCCL_TMA_SLOT_SIZE);
          // tma_barrier_t* tmaBar = reinterpret_cast<tma_barrier_t*>(&barrierStorage[tmaSlot]);
          
          if (Recv) { // Recv and RecvSend
            for (int i = 0; i < fan.nrecv(); i++) {
              if (ncclShmem.groups[group].srcs[i] != nullptr) {
                void* globalSrc = ncclShmem.groups[group].srcs[i];
                void* shmemDst = (char*)tmaShmemSlot + i * currentSliceSize * sizeof(T);
                size_t loadBytes = currentSliceSize * sizeof(T);
                
                TMA_DEBUG_PRINT("TMA PRELOAD[%d]: slot=%d, globalSrc=%p, shmemDst=%p, loadBytes=%lu", 
                                preloadIdx, tmaSlot, globalSrc, shmemDst, (unsigned long)loadBytes);
                
                #if __CUDA_ARCH__ >= 900
                TMA_DEBUG_PRINT("TMA PRELOAD[%d]: Calling cuda::memcpy_async (ARCH=%d)", preloadIdx, __CUDA_ARCH__);
                cuda::memcpy_async(
                  shmemDst,
                  reinterpret_cast<const unsigned char*>(globalSrc),
                  cuda::aligned_size_t<16>(loadBytes),
                  *tmaBar);
                TMA_DEBUG_PRINT("TMA PRELOAD[%d]: cuda::memcpy_async returned", preloadIdx);
                #else
                TMA_DEBUG_PRINT("TMA PRELOAD[%d]: SKIPPED - ARCH=%d < 900", preloadIdx, __CUDA_ARCH__);
                #endif
              }
            }
          } else if (Send && Src) { // Send only
            if (ncclShmem.groups[group].srcs[0] != nullptr) {
              void* globalSrc = ncclShmem.groups[group].srcs[0];
              void* shmemDst = (char*)tmaShmemSlot;
              size_t loadBytes = currentSliceSize * sizeof(T);
              
              TMA_DEBUG_PRINT("TMA PRELOAD[%d]: slot=%d, globalSrc=%p, loadBytes=%lu", 
                              preloadIdx, tmaSlot, globalSrc, (unsigned long)loadBytes);
              
              #if __CUDA_ARCH__ >= 900
              TMA_DEBUG_PRINT("TMA PRELOAD[%d] SEND: Calling cuda::memcpy_async (ARCH=%d)", preloadIdx, __CUDA_ARCH__);
              cuda::memcpy_async(
                shmemDst,
                reinterpret_cast<const unsigned char*>(globalSrc),
                cuda::aligned_size_t<16>(loadBytes),
                *tmaBar);
              TMA_DEBUG_PRINT("TMA PRELOAD[%d] SEND: cuda::memcpy_async returned", preloadIdx);
              #else
              TMA_DEBUG_PRINT("TMA PRELOAD[%d] SEND: SKIPPED - ARCH=%d < 900", preloadIdx, __CUDA_ARCH__);
              #endif
            }
          }
        }
        
        // All threads arrive at barrier and save token
        // tma_barrier_t* tmaBar = reinterpret_cast<tma_barrier_t*>(&barrierStorage[tmaSlot]);
        tmaTokens[tmaSlot] = tmaBar->arrive();
        
        offset += currentSliceSize;
      }
      
      // ========================================
      // MAIN LOOP: Consume 1, Produce 1
      // ========================================
      offset = 0; // Reset offset for main loop
      for (int slice = 0; slice < totalSlices; ++slice) {
        int currentSliceSize = sliceSize < nelem-offset ? sliceSize : nelem-offset;
        int tmaSlot = slice % NCCL_TMA_PIPE_DEPTH;
        
        // CONSUME: Even though we already loaded data in prologue/produce,
        // we still need to call waitPeer to synchronize step across all threads
        // (RoleWaitRecv, RolePostRecv, etc. need to have consistent step values)
        if (tid == 0) {
          T* userInput = (T*)ncclShmem.groups[group].userInput;
          T* userOutput = (T*)ncclShmem.groups[group].userOutput;
          if (Src) ncclShmem.groups[group].srcs[0] = (SrcBuf==Input ? userInput : userOutput) + srcIx + offset;
          if (Dst) ncclShmem.groups[group].dsts[0] = (DstBuf==Input ? userInput : userOutput) + dstIx + offset;
        }
        waitPeer<DirectRecv, DirectSend, Recv, Send, Src, Dst>(srcIx, dstIx, offset, currentSliceSize);
        
        TMA_DEBUG_PRINT("TMA MAIN LOOP[%d]: After waitPeer, step=%ld", slice, step);
        
        // Wait for TMA load completion of current slice
        tma_barrier_t* tmaBar = reinterpret_cast<tma_barrier_t*>(&barrierStorage[tmaSlot]);
        tmaBar->wait(std::move(tmaTokens[tmaSlot]));
        
        TMA_DEBUG_PRINT("TMA WAIT DONE: slice=%d, tmaSlot=%d", slice, tmaSlot);
        
        // Debug: Check SMEM right after wait completes
        if (tid == 0) {
          void* tmaShmemSlot = ncclTmaShmemSlot(tmaSlot, NCCL_TMA_SLOT_SIZE);
          T* smemData = (T*)tmaShmemSlot;
          TMA_DEBUG_PRINT("  AFTER WAIT: smemData[0]=%f, smemData[1]=%f", 
                          (float)smemData[0], (float)smemData[1]);
        }
        
        // [Optimization] Move postPeer earlier to unblock Sender
        // logic:
        // 1. Recv Only: We have data in SMEM. We can release the GMEM buffer (Tail) immediately.
        // 2. Send Only / RecvSend: We must WRITE data to GMEM (FIFO) before signaling (Head).
        //    So we must wait until reduceCopyFromSmem completes.
        bool earlyPost = Recv && !Send;
        if (earlyPost) {
           barrier(); // Ensure all threads see data arrival
           postPeer<Recv, Send>(0 < currentSliceSize);
        }

        // SECOND: Update src pointers to point to SMEM (where TMA loaded the data)
        if (tid == 0) {
          void* tmaShmemSlot = ncclTmaShmemSlot(tmaSlot, NCCL_TMA_SLOT_SIZE);
          if (Recv) {
            for (int i = 0; i < fan.nrecv(); i++) {
              if (ncclShmem.groups[group].srcs[i] != nullptr) {
                void* oldSrc = ncclShmem.groups[group].srcs[i];
                ncclShmem.groups[group].srcs[i] = (T*)((char*)tmaShmemSlot + i * currentSliceSize * sizeof(T));
                TMA_DEBUG_PRINT("TMA UPDATE SRC[%d]: %p -> %p (SMEM)", i, oldSrc, ncclShmem.groups[group].srcs[i]);
              }
            }
          } else if (Send && Src) {
            void* oldSrc = ncclShmem.groups[group].srcs[0];
            ncclShmem.groups[group].srcs[0] = (T*)tmaShmemSlot;
            TMA_DEBUG_PRINT("TMA UPDATE SRC[0]: %p -> %p (SMEM)", oldSrc, ncclShmem.groups[group].srcs[0]);
          }
        }
        
        subBarrier();
        
        // ========================================
        // PROCESS: ReduceCopy current slice (uses data from SMEM)
        // ========================================
        int workSize = ncclShmem.aborted ? 0 : currentSliceSize;
        bool usedTmaLoad = true;  // Always true in TMA path
        
        if (flags & AnyNetDeviceUnpack) {
          ncclNetDeviceUnpack<Recv>(tid, tidInBlock, nworkers, group, ncclShmem.groups[group].devicePlugin.unpack.unpackNetDeviceIndexMask, Src, workSize);
          subBarrier();
        }

        // Perform reduceCopy from SMEM
        if (ncclShmem.groups[group].srcs[0] && ncclShmem.groups[group].dsts[0]) {
          constexpr int PreOpSrcs = SrcBuf != Input ? 0 :
                                    DirectRecv*MaxRecv == NCCL_MAX_DIRECT_ARITY ? (1+NCCL_MAX_DIRECT_ARITY) : 1;
          
          TMA_DEBUG_PRINT("REDUCE_COPY (TMA): slice=%d, workSize=%d", slice, workSize);
          
          reduceCopyFromSmem<Unroll, RedOp, T,
            MultimemDsts, Send + Dst, Send * MaxSend + Dst, PreOpSrcs>
            (tid, nworkers, ncclShmem.redOpArgs[0], ncclShmem.redOpArgs, false /*postOp*/,
              Recv * fan.nrecv() + Src, [&](int i) { return ncclShmem.groups[group].srcs[i]; },
              Send * fan.nsend() + Dst, [&](int i) { return ncclShmem.groups[group].dsts[i]; },
              workSize);
        } else {
          workSize = 0;
        }
        TMA_DEBUG_PRINT("BEFORE POST PEER (TMA): slice=%d", slice);
        if (!earlyPost) {
           barrier(); // Ensure reduceCopyFromSmem completed writing to GMEM
           postPeer<Recv, Send>(0 < workSize);
        }
        TMA_DEBUG_PRINT("AFTER POST PEER (TMA): slice=%d", slice);
        // ========================================
        // PRODUCE: Issue TMA load for next slice (slice + NCCL_TMA_PIPE_DEPTH)
        // ========================================
        int nextSlice = slice + NCCL_TMA_PIPE_DEPTH;
        if (nextSlice < totalSlices) {
          int nextOffset = offset + NCCL_TMA_PIPE_DEPTH * sliceSize;
          int nextSliceSize = sliceSize < nelem-nextOffset ? sliceSize : nelem-nextOffset;
          int nextTmaSlot = nextSlice % NCCL_TMA_PIPE_DEPTH;
          
          // For Send operations, set source pointer to user buffer first
          if (tid == 0 && Send && Src) {
            T* userInput = (T*)ncclShmem.groups[group].userInput;
            T* userOutput = (T*)ncclShmem.groups[group].userOutput;
            ncclShmem.groups[group].srcs[0] = (SrcBuf==Input ? userInput : userOutput) + srcIx + nextOffset;
          }
          
          // [TMA PRODUCE] Wait for peer for the next slice
          // Pass relative offset from current step
          // Note: Since postPeer acts early, step has been incremented.
          // WaitPeer (in loop start) + PostPeer (moved up) = Step + 2*StepPerSlice.
          // Before change: PostPeer was at end. Step was +1*StepPerSlice when waitPeerForTmaLoad called? No.
          // waitPeer() + postPeer() are both called inside loop.
          // Before: waitPeer() -> reduce() -> postPeer() -> produce()
          // Produce called when step was already incremented by both (if produce was after postPeer?)
          // Wait, Produce is at end of loop.
          // Original: waitPeer -> reduce -> barrier -> postPeer -> produce
          // So step was incremented by TWO.
          // Next Slice Relative Offset: nextSlice - (slice + 1).
          // If step corresponds to Slice+1 (completed), then nextSlice is distant by (nextSlice - (Slice+1)).
          // Code logic remains same.
          waitPeerForTmaLoad<DirectRecv, DirectSend, Recv, Send, Src, Dst>(
            srcIx, dstIx, nextOffset, nextSliceSize, /*sliceStep=*/nextSlice - (slice + 1));
          
          if (tid == 0) {
            TMA_DEBUG_PRINT("TMA PRODUCE[%d]: step=%ld srcPtr=%p", 
                            nextSlice, step, ncclShmem.groups[group].srcs[0]);
          }
          subBarrier();
          
          // Issue TMA for next slice
          if (tid == 0) {
            void* tmaShmemSlot = ncclTmaShmemSlot(nextTmaSlot, NCCL_TMA_SLOT_SIZE);
            tma_barrier_t* tmaBar = reinterpret_cast<tma_barrier_t*>(&barrierStorage[nextTmaSlot]);
            
            if (Recv) {
              for (int i = 0; i < fan.nrecv(); i++) {
                if (ncclShmem.groups[group].srcs[i] != nullptr) {
                  void* globalSrc = ncclShmem.groups[group].srcs[i];
                  void* shmemDst = (char*)tmaShmemSlot + i * nextSliceSize * sizeof(T);
                  size_t loadBytes = nextSliceSize * sizeof(T);
                  
                  TMA_DEBUG_PRINT("TMA PRODUCE[%d]: slot=%d, globalSrc=%p, loadBytes=%lu", 
                                  nextSlice, nextTmaSlot, globalSrc, (unsigned long)loadBytes);
                  
                  #if __CUDA_ARCH__ >= 900
                  cuda::memcpy_async(
                    shmemDst,
                    reinterpret_cast<const unsigned char*>(globalSrc),
                    cuda::aligned_size_t<16>(loadBytes),
                    *tmaBar);
                  #endif
                }
              }
            } else if (Send && Src) {
              if (ncclShmem.groups[group].srcs[0] != nullptr) {
                void* globalSrc = ncclShmem.groups[group].srcs[0];
                void* shmemDst = (char*)tmaShmemSlot;
                size_t loadBytes = nextSliceSize * sizeof(T);
                
                TMA_DEBUG_PRINT("TMA PRODUCE[%d]: slot=%d, loadBytes=%lu", 
                                nextSlice, nextTmaSlot, (unsigned long)loadBytes);
                
                #if __CUDA_ARCH__ >= 900
                cuda::memcpy_async(
                  shmemDst,
                  reinterpret_cast<const unsigned char*>(globalSrc),
                  cuda::aligned_size_t<16>(loadBytes),
                  *tmaBar);
                #endif
              }
            }
          }
          
          // Save token for next slice
          tma_barrier_t* tmaBar = reinterpret_cast<tma_barrier_t*>(&barrierStorage[nextTmaSlot]);
          tmaTokens[nextTmaSlot] = tmaBar->arrive();
        }
        
        offset += currentSliceSize;
      }
      
      // TMA path done
    } else if (tid < nworkers && nelem > 0 && !isNetOffload) {
      // ========================================
      // NON-TMA PATH (Fallback)
      // ========================================
      int slice = 0;
      int offset = 0;
      
      #if __CUDA_ARCH__ < 700
        #pragma unroll SlicePerChunk
      #else
        #pragma unroll 1
      #endif
      do {
        int currentSliceSize = sliceSize < nelem-offset ? sliceSize : nelem-offset;
        if (tid == 0) {
          T* userInput = (T*)ncclShmem.groups[group].userInput;
          T* userOutput = (T*)ncclShmem.groups[group].userOutput;
          if (Src) ncclShmem.groups[group].srcs[0] = (SrcBuf==Input ? userInput : userOutput) + srcIx + offset;
          if (Dst) ncclShmem.groups[group].dsts[0] = (DstBuf==Input ? userInput : userOutput) + dstIx + offset;
        }
        waitPeer<DirectRecv, DirectSend, Recv, Send, Src, Dst>(srcIx, dstIx, offset, currentSliceSize);
        
        subBarrier();
        int workSize = ncclShmem.aborted ? 0 : currentSliceSize;
        
        if (ncclShmem.groups[group].srcs[0] && ncclShmem.groups[group].dsts[0]) {
          constexpr int PreOpSrcs = SrcBuf != Input ? 0 :
                                    DirectRecv*MaxRecv == NCCL_MAX_DIRECT_ARITY ? (1+NCCL_MAX_DIRECT_ARITY) : 1;
          
          reduceCopy<Unroll, RedOp, T,
            MultimemSrcs, Recv + Src, Recv * MaxRecv + Src,
            MultimemDsts, Send + Dst, Send * MaxSend + Dst, PreOpSrcs>
            (tid, nworkers, ncclShmem.redOpArgs[0], ncclShmem.redOpArgs, postOp,
              Recv * fan.nrecv() + Src, ncclShmem.groups[group].srcs,
              Send * fan.nsend() + Dst, ncclShmem.groups[group].dsts,
              workSize);
        } else {
          workSize = 0;
        }
        barrier();
        postPeer<Recv, Send>(0 < workSize);
        offset += currentSliceSize;
        slice += 1;
      } while (slice < SlicePerChunk && offset < nelem);
    }

    // Non-workers or empty slices
    // int slice = shouldUseTma ? totalSlices : 0;
    int slice = tid < nworkers ? totalSlices : 0;
    #pragma unroll 1
    while (slice < SlicePerChunk) {
      int offset = slice * sliceSize;
      int currentSliceSize = sliceSize < nelem-offset ? sliceSize : nelem-offset;
      { waitPeer<DirectRecv, DirectSend, Recv, Send, Src, Dst>(0, 0, 0, currentSliceSize); }
      barrier();
      int workSize = ncclShmem.aborted ? 0 : currentSliceSize;
      postPeer<Recv, Send>(0 < workSize);
      slice += 1;
    }
  }

public:
  static inline __device__ void sendPeerNotify(int peer, int connIndex, int steps) {
    ncclDevChannelPeer* peerPtr = ncclShmem.channel.peers[peer];
    peerPtr->send[connIndex].step += steps;
    st_relaxed_sys_global(peerPtr->send[connIndex].tail, peerPtr->send[connIndex].step);
  }

  static inline __device__ void recvPeerNotify(int peer, int connIndex, int steps) {
    int spins = 0;
    ncclDevChannelPeer* peerPtr = ncclShmem.channel.peers[peer];
    peerPtr->recv[connIndex].step += steps;
    st_relaxed_sys_global(peerPtr->recv[connIndex].head, peerPtr->recv[connIndex].step);
    while (ld_volatile_global(peerPtr->recv[connIndex].tail) < peerPtr->recv[connIndex].step) {
      int abort = 0;
      if (checkAbort(abort, 1, spins)) break;
    }
  }

  template<int Recv, int Send, typename Fn>
  __device__ __forceinline__ void process(Fn &&fn, uint32_t sendDirectFlag = 0, uint32_t recvDirectFlag = 0) {
    #pragma unroll 1
    for (int slice=0; slice < SlicePerChunk; slice++) {
      if (tid < nworkers) {
        int nsend, nrecv;
        if (flags & (Recv*RoleWaitRecv | Send*RoleWaitSend)) {
          const bool isSendNotRecv = (Send && Recv) ? (flags & RoleWaitSend) : Send;
          int spins = 0;
          while (connStepCache + (isSendNotRecv ? NCCL_STEPS : 0) < step + StepPerSlice) {
            connStepCache = loadStepValue(connStepPtr);
            if (checkAbort(flags, Aborted, spins)) break;
          }
          void **ptrs = isSendNotRecv ? ncclShmem.groups[group].dsts
                                      : ncclShmem.groups[group].srcs;
          if ((flags & ConnFifoEnabled) && connFifo[step%NCCL_STEPS].mode == NCCL_MODE_OFFSET) {
            int offset = loadInt(&connFifo[step%NCCL_STEPS].offset);
            ptrs[index] = connEltsFifo + offset/sizeof(T);
          } else if (Direct && fn.work->regUsed) {
            if (isSendNotRecv) {
              if (flags & DirectWrite) {
                ptrs[index] = directBuff;
              } else if (flags & DirectRead) {  // empty send
                ptrs[index] = nullptr;
              } else {
                ptrs[index] = connEltsFifo + (step%NCCL_STEPS)*connStepSize;
              }
            } else {
              if (flags & DirectRead) {
                ptrs[index] = directBuff;
              } else if (flags & DirectWrite) {
                if (Send)
                  ptrs[index] = directBuff;  // send to next from my output buffer
                else
                  ptrs[index] = nullptr;
              } else {
                ptrs[index] = connEltsFifo + (step%NCCL_STEPS)*connStepSize;
              }
            }
          } else {
            ptrs[index] = connEltsFifo + (step%NCCL_STEPS)*connStepSize;
          }
        }
        subBarrier();
        if (Recv == 0 || ncclShmem.groups[group].srcs[0] == nullptr) {
          nrecv = 0;
        } else {
          nrecv = fan.nrecv();
        }

        if (Send == 0 || ncclShmem.groups[group].dsts[0] == nullptr) {
          nsend = 0;
        } else {
          nsend = fan.nsend();
        }
        fn.template operator()<SlicePerChunk, 0, Recv*MaxRecv, 0, Send*MaxSend, MultimemSrcs, MultimemDsts>
          (tid, nworkers, slice, stepSize * StepPerSlice,
            nrecv, ncclShmem.groups[group].srcs,
            nsend, ncclShmem.groups[group].dsts, ncclShmem.groups[group].dstSizes, sendDirectFlag, recvDirectFlag);
      }
      barrier();
      int32_t dstSize = 0;
      if (flags & Send*RolePostSend) {
        // Yes, for some template arguments this code will be unreachable.  That's fine.
        // coverity[dead_error_begin]
        dstSize = ncclShmem.groups[group].dstSizes[index];
        ncclShmem.groups[group].dstSizes[index] = 0;
        if (flags & ConnFifoEnabled) connFifo[step%NCCL_STEPS].size = dstSize*sizeof(T);
      }
      barrier();
      if (flags & (Recv*(RoleWaitRecv|RolePostRecv) | Send*(RoleWaitSend|RolePostSend))) {
        step += StepPerSlice;
      }
      if (flags & (Recv*RolePostRecv | Send*RolePostSend)) {
        if (Send && (!Recv || (flags & RolePostSend)) && (dstSize!=0 || (flags&ConnFifoEnabled))) {
          fence_acq_rel_sys();
        }
        st_relaxed_sys_global(connStepPtr, step);
      }
    }
  }

private:
  // Scatter/Gather generic op
  // skip: my own rank order in the buffer chunks
  // shift: peer offset to avoid all ranks sending to or receiving from same peer
  template <int DirectRecv1, int DirectSend1, int Recv, int Send>
  __device__ __forceinline__ void
  ScatterGatherOp(intptr_t inpIx, intptr_t outIx, ssize_t totalElem, int peerElem, ssize_t peerOffset, int skip, int shift, bool postOp) {
    constexpr int DirectRecv = 1 && Direct && DirectRecv1;
    constexpr int DirectSend = 1 && Direct && DirectSend1;
    int offset = 0; // slice offset
    int sliceSize = stepSize*StepPerSlice;
    int dataSize = max(DIVUP(peerElem, 16*SlicePerChunk)*16, sliceSize/32);  // per-peer slice size

    #pragma unroll
    for (int slice=0; slice<SlicePerChunk; ++slice) {
      ssize_t realSize = max(0, min(dataSize, peerElem-offset));
      bool fenceNeeded = false;
      if (tid < nworkers) {
        if (Send) {
          // Scatter pre-scales data of input buffer only in non-Direct case
          constexpr int PreOpSrcs = DirectSend ? 0 : 1;
          if (tid==0) ncclShmem.groups[group].srcs[0] = (T*)ncclShmem.groups[group].userInput + inpIx + offset;
          // realSize is not accurate here; but intra-node does not rely on sizes FIFO
          waitPeer<0, DirectSend, 0, 1, 1, 0>(0, inpIx, offset, realSize);
          subBarrier();
          #pragma unroll
          // Loop over peers
          for (int j=0; j<fan.nsend(); j++) {
            int i = (j+shift)%fan.nsend();
            ssize_t pOffset = i*peerOffset;
            // Skip the data I am responsible of reducing myself
            if (skip >= 0 && i >= skip) pOffset += peerOffset;
            void* src0 = (T*)ncclShmem.groups[group].srcs[0] + pOffset;
            ssize_t realPeerSize = min(realSize, totalElem-pOffset);
            if (realPeerSize > 0 && ncclShmem.groups[group].dsts[i] != nullptr) {
              reduceCopy<Unroll, RedOp, T, 0,1,1, 0,1,1, PreOpSrcs>(tid, nworkers, ncclShmem.redOpArgs[0], ncclShmem.redOpArgs, false, 1, &src0, 1, ncclShmem.groups[group].dsts+i, realPeerSize);
              // Mark for threadfence at the end
              fenceNeeded |= true;
            }
          }
        } else if (Recv) {
          if (tid==0) ncclShmem.groups[group].dsts[0] = (T*)ncclShmem.groups[group].userOutput + outIx + offset;
          ssize_t pOffset = index*peerOffset;
          if (skip >= 0 && index >= skip) pOffset += peerOffset;
          // Adjust remote index with peer offset in case we are directly pulling from peer's output buffer
          waitPeer<DirectRecv, 0, 1, 0, 0, 1>(outIx+pOffset, outIx+pOffset, offset, realSize);
          subBarrier();
          #pragma unroll
          for (int j=0; j<fan.nrecv(); j++) {
            int i = (j+shift)%fan.nrecv();
            pOffset = i*peerOffset;
            if (skip >= 0 && i >= skip) pOffset += peerOffset;
            void* dst0 = (T*)ncclShmem.groups[group].dsts[0] + pOffset;
            ssize_t realPeerSize = min(realSize, totalElem-pOffset);
            if (DirectRecv && ncclShmem.groups[group].srcs[i] == dst0) realPeerSize = 0;
            if (realPeerSize > 0) reduceCopy<Unroll, RedOp, T, 0,1,1, 0,1,1, /*PreOpSrcs=*/0>(tid, nworkers, ncclShmem.redOpArgs[0], ncclShmem.redOpArgs, postOp, 1, ncclShmem.groups[group].srcs+i, 1, &dst0, realPeerSize);
          }
        }
      }
      fenceNeeded = barrierAny(fenceNeeded);
      postPeer<Recv, Send>(fenceNeeded);
      offset += realSize;
    }
  }

  __device__ __forceinline__ void loadRecvConn(ncclDevChannelPeer *peer, int connIndex, uint32_t direct, int ipcRegFlag, int netRegFlag) {
    conn = &peer->recv[connIndex];
    if (conn->netDeviceHandle.netDeviceType == NCCL_NET_DEVICE_UNPACK) {
      // handle must be a device ptr
      netDeviceHandle = conn->netDeviceHandle.handle;
      // Cache the handle
      ncclNetDeviceUnpackSetup(netDeviceHandle, group, index);
      flags |= NetDeviceUnpack;
    }
    step = conn->step;
    step = roundUp(step, SlicePerChunk*StepPerSlice);
    if (flags & RolePostRecv) {
      connStepPtr = conn->head;
      *connStepPtr = step; // Return credits in case we rounded up.
    }
    if (flags & RoleWaitRecv) {
      if ((flags & PatMode) == 0) ncclShmem.groups[group].recvConns[index] = conn; // WaitRecv role saves since that's who needs it in setDataPtrs()
      flags |= (conn->flags & NCCL_NVLS_MIN_POLL) ? NvlsMinPolling : 0;
      connStepPtr = conn->tail;
      connStepCache = loadStepValue(connStepPtr);
      connStepSize = conn->stepSizes[NCCL_PROTO_TMA]/sizeof(T);
      connEltsFifo = (T*)conn->buffs[NCCL_PROTO_TMA];
      if (conn->connFifo != nullptr) {
        flags |= ConnFifoEnabled;
        connFifo = conn->connFifo;
      }
      if (Direct) {
        if (ipcRegFlag) {
          // User buffers have been registered
          if (conn->flags & (NCCL_P2P_READ | NCCL_P2P_WRITE)) {
            if (P2p) {
              flags |= conn->flags & NCCL_P2P_WRITE ? DirectWrite : DirectRead;
            } else if (connIndex == 1 && direct) {
              flags |= DirectRead;
            } else {
              flags |= direct & NCCL_P2P_READ ? DirectRead : DirectWrite;
            }
          } else if ((conn->flags & NCCL_NVLS_MIN_POLL)) {
            /* NVLS direct */
            flags |= DirectRead;
          }
        }
        if (netRegFlag) {
          if (conn->flags & NCCL_DIRECT_NIC) {
            flags |= NetRegMode;
            connFifo[step % NCCL_STEPS].size = 0;
          }
        }
      }
    }
  }

  __device__ __forceinline__ void loadSendConn(ncclDevChannelPeer *peer, int connIndex, uint32_t direct, int ipcRegFlag, int netRegFlag) {
    conn = &peer->send[connIndex];
    step = conn->step;
    step = roundUp(step, SlicePerChunk*StepPerSlice);

    connFifo = conn->connFifo;
    if (connFifo != nullptr) flags |= ConnFifoEnabled;

    if (flags & RolePostSend) {
      connStepPtr = conn->tail;
      connEltsFifo = (T*)conn->buffs[NCCL_PROTO_TMA];
    }
    if (flags & RoleWaitSend) {
      if ((flags & PatMode) == 0) ncclShmem.groups[group].sendConns[index] = conn; // WaitSend role saves since that's who needs it in setDataPtrs()
      flags |= (conn->flags & NCCL_NVLS_MIN_POLL) ? NvlsMinPolling : 0;
      connStepPtr = conn->head;
      connStepCache = loadStepValue(connStepPtr);
      connStepSize = conn->stepSizes[NCCL_PROTO_TMA]/sizeof(T);
      connEltsFifo = (T*)conn->buffs[NCCL_PROTO_TMA];
      if (Direct) {
        if (ipcRegFlag) {
          // User buffers have been registered
          if (conn->flags & (NCCL_P2P_WRITE | NCCL_P2P_READ)) {
            if (P2p) {
              flags |= conn->flags & NCCL_P2P_WRITE ? DirectWrite : DirectRead;
            } else if (connIndex == 1 && direct) {
              flags |= DirectRead;  // scatter-reduce use direct pull
            } else {
              flags |= direct & NCCL_P2P_READ ? DirectRead : DirectWrite;
            }
          } else if ((conn->flags & NCCL_NVLS_MIN_POLL)) {
            /* NVLS direct */
            flags |= DirectWrite;
          }
        }
        if (netRegFlag) {
          if (conn->flags & NCCL_DIRECT_NIC) {
            flags |= NetRegMode;
          }
        }
      }
    }
  }

 public:
  __device__ Primitives(
      int tid, int nthreads, int const *recvPeers, int const *sendPeers,
      void const *inputBuf, void *outputBuf, uint64_t redOpArg, uint8_t group=0,
      uint8_t connIndexRecv = 0, uint8_t connIndexSend = 0, struct ncclDevWorkColl* collWork = nullptr,
      struct ncclDevWorkP2p* p2pWork = nullptr, int stepSize_ = 0, int mode = primsModeDefault
    ):
    tid(tid), nthreads(nthreads), tidInBlock(threadIdx.x), group(group),
    stepSize(stepSize_ == 0 ? ncclShmem.comm.buffSizes[NCCL_PROTO_TMA]/NCCL_STEPS/sizeof(T) : stepSize_) {

    int peer = -1;
    flags = 0;
    index = -1;
    if (mode == primsModeDefault) { // Connect to ranks in sendPeers/recvPeers
      // For send operations, we need an extra warp to overlap the threadfence and the copy
      this->nworkers = nthreads - (MaxSend > 0 && nthreads >= NCCL_SIMPLE_EXTRA_GROUP_IF_NTHREADS_GE ? WARP_SIZE : 0);

      int nrecv=0, nsend=0;
      // Yes, for some template arguments this code will be unreachable.  That's fine.
      // coverity[dead_error_line]
      while (nrecv < MaxRecv && recvPeers[nrecv] != -1) nrecv++;
      // coverity[dead_error_line]
      while (nsend < MaxSend && sendPeers[nsend] != -1) nsend++;
      this->fan = Fan(nrecv, nsend);

      constexpr int ThreadPerSync =
        MaxSend >= 16 || MaxRecv >= 16 ? 32 : // NVLS may have an arity > 8. In that case increase the size of the groups
        MaxSend >= 8 || MaxRecv >= 8 ? 16 :
        8; // Allows for all roles (WaitRecv/WaitSend/PostRecv/PostSend) within a single warp
      static_assert(MaxSend <= ThreadPerSync && MaxRecv <= ThreadPerSync, "Not enough threads to cover all peers");

      assert(2*(nrecv+nsend) <= nthreads); // Ensure no thread is assigned more than one role.
      // Coverity assumes that index will equal tid based on the line below, but it doesn't consider the setting
      // of flags.  This results in multiple false positive overruns being reported here and in all_reduce.h.
      // Unfortunately, we've been unsuccessful in trying to silence them with a single directive here so
      // instead it's being done at the callers.
      // coverity[assignment:FALSE]
      if      (tid < nrecv)                 { flags |= RoleWaitRecv; index = tid; }
      // Yes, for some template arguments this code will be unreachable.  That's fine.
      // coverity[dead_error_begin]
      else if (tid < nrecv+nsend)           { flags |= RoleWaitSend; index = tid-nrecv; }
      else if (nthreads-nsend <= tid)       { flags |= RolePostSend; index = tid-(nthreads-nsend); }
      else if (nthreads-nrecv-nsend <= tid) { flags |= RolePostRecv; index = tid-(nthreads-nrecv-nsend); }

      if (flags & (RoleWaitRecv|RolePostRecv)) peer = recvPeers[index];
      if (flags & (RoleWaitSend|RolePostSend)) peer = sendPeers[index];

      // Coverity thinks that index could be -1 here but that's not actually the case.
      // coverity[negative_returns:FALSE]
      int sendIpcReg;
      int recvIpcReg;
      int sendNetReg;
      int recvNetReg;
      if (P2p) {
        sendIpcReg = p2pWork ? p2pWork->sendIpcReg : 0;
        recvIpcReg = p2pWork ? p2pWork->recvIpcReg : 0;
        sendNetReg = p2pWork ? p2pWork->sendNetReg : 0;
        recvNetReg = p2pWork ? p2pWork->recvNetReg : 0;
      } else {
        recvIpcReg = sendIpcReg = collWork ? collWork->regUsed : 0;
        recvNetReg = sendNetReg = collWork ? collWork->netRegUsed : 0;
      }

      // coverity[overrun-call] => Coverity think prims.index can be greater than 1
      if (flags & (RoleWaitRecv|RolePostRecv)) loadRecvConn(ncclShmem.channel.peers[peer], connIndexRecv, collWork ? collWork->direct : 0, recvIpcReg, recvNetReg);
      // coverity[overrun-call] => Coverity think prims.index can be greater than 1
      if (flags & (RoleWaitSend|RolePostSend)) loadSendConn(ncclShmem.channel.peers[peer], connIndexSend, collWork ? collWork->direct : 0, sendIpcReg, sendNetReg);

      if (barrierAny(flags & NetDeviceUnpack)) {
        flags |= AnyNetDeviceUnpack;
        // RoleWaitRecv starts at tid=0, so this creates the bitmask of which recv peers
        // have NetDeviceUnpack.
        uint32_t mask = __ballot_sync(~0u, ((flags & RoleWaitRecv) && (flags & NetDeviceUnpack)) ? 1 : 0);
        if (tid == 0) {
          ncclShmem.groups[this->group].devicePlugin.unpack.unpackNetDeviceIndexMask = mask;
        }
      }

      // coverity[negative_returns:FALSE] => coverity thinks that index could be -1 but that's not actually the case
      // coverity[var_deref_model] => coverity thinks work can dereferenced if NULL but this is not the case
      setDataPtrs(inputBuf, outputBuf, redOpArg, (struct ncclDevWorkCollReg*)collWork, sendIpcReg || recvIpcReg, peer);
      // coverity[uninit_member] => coverity thinks fan.n is not initialized
    } else if (mode == primsModePatRs || mode == primsModePatAg) { // Connect to all ranks +/- 2^n
      flags |= PatMode;
      const int roles[5] = { RoleWaitRecv, RolePostRecv, RoleWaitSend, RolePostSend, RoleInput | RoleOutput };
      if (tid < 5) flags |= roles[tid];

      int nranks = ncclShmem.comm.nRanks;
      if (tid < 32 && ((1UL<<tid) < nranks)) {
        int rank = ncclShmem.comm.rank;
        uint32_t delta = 1 << tid;
        // Load recv peer
        int recvPeer = mode == primsModePatRs ? (rank - delta + nranks) % nranks : (rank + delta) % nranks;
        struct ncclPatPeer* peer = ((struct ncclPatPeer*)recvPeers)+tid;
        struct ncclConnInfo* conn = peer->conn = ncclShmem.channel.peers[recvPeer]->recv+connIndexRecv;
        peer->step = conn->step;
        peer->buff = conn->buffs[NCCL_PROTO_TMA];
        peer->stepCache = loadStepValue(peer->tailPtr = conn->tail);
        peer->headPtr = conn->head;
        peer->accSize = 0;
        peer->connStepSize = conn->stepSizes[NCCL_PROTO_TMA]/sizeof(T);
        // Load send peer
        int sendPeer = mode == primsModePatAg ? (rank - delta + nranks) % nranks : (rank + delta) % nranks;
        peer = ((struct ncclPatPeer*)sendPeers)+tid;
        conn = peer->conn = ncclShmem.channel.peers[sendPeer]->send+connIndexSend;
        peer->step = conn->step;
        peer->connFifo = conn->connFifo;
        peer->buff = conn->buffs[NCCL_PROTO_TMA];
        peer->stepCache = loadStepValue(peer->headPtr = conn->head);
        peer->tailPtr = conn->tail;
        peer->accSize = 0;
        peer->connStepSize = conn->stepSizes[NCCL_PROTO_TMA]/sizeof(T);
      }
      if (tid==0) {
        ncclShmem.groups[group].userInput = (void*)inputBuf;
        ncclShmem.groups[group].userOutput = (void*)outputBuf;
        ncclShmem.redOpArgs[0] = redOpArg;  // scaler for local input
      }
      patBarrier();
    }
  }

  __device__ ~Primitives() {
    if (flags&PatMode) return;
    // Save steps for the next operation
    if (flags & (RolePostSend|RolePostRecv)) conn->step = step;
    if ((flags & NetRegMode) && (flags & RoleWaitSend)) {
      // Make sure we wait until the proxy has sent data before we return.
      // We don't want the next CUDA kernel to overwrite the send buffer which
      // was accessed directly.
      uint64_t prevStep = step - StepPerSlice;
      volatile ssize_t* ptr = &(connFifo[prevStep%NCCL_STEPS].size);
      int spins = 0;
      while (*ptr != -1) if (checkAbort(flags, Aborted, spins)) break;
    }

    if (flags & NetDeviceUnpack) {
      ncclNetDeviceSaveHead(netDeviceHandle, group, index);
    }

    // Make sure all threads are done writing back conn->step and done using
    // ncclShmem.groups[group]
    barrier();

    if ((flags & DirectRead) && (flags & RoleWaitSend) && P2p) {
      // For sendrecv DirectRead, sender needs to wait for receiver reading data from src.
      // This has to be done after barrier() since post thread might have contention with
      // this check.
      int spins = 0;
      volatile uint64_t* tail = conn->tail;
      volatile uint64_t* head = conn->head;
      while (*tail > *head) if (checkAbort(flags, Aborted, spins)) break;
    }
  }

  __device__ void setDataPtrs(void const *inputBuf, void *outputBuf, uint64_t redOpArg, struct ncclDevWorkCollReg* work, uint8_t ipcReg, int peer) {
    if (tid==0) {
      ncclShmem.groups[group].userInput = (void*)inputBuf;
      ncclShmem.groups[group].userOutput = (void*)outputBuf;
      ncclShmem.redOpArgs[0] = redOpArg;  // scaler for local input
    }

    if (Direct && ipcReg) {
      bool recvProvider = (flags & RoleWaitRecv) && (flags & DirectWrite);
      bool sendAcceptor = (flags & RoleWaitSend) && (flags & DirectWrite);
      bool sendProvider = (flags & RoleWaitSend) && (flags & DirectRead); // sender provides direct buffer (to be fetched)
      bool recvAcceptor = (flags & RoleWaitRecv) && (flags & DirectRead); // receiver accepts direct buffer
      if (recvProvider) {
        int spins = 0;
        void* volatile* slot = ncclShmem.groups[group].recvConns[index]->ptrExchange;
        // Wait for consumer to consume previous value before trampling it.
        if (slot) {
          T* exchgPtr;
          directBuff = (T*)outputBuf;
          while (*slot != nullptr && !checkAbort(flags, Aborted, spins));
          if (P2p) {
            exchgPtr = (T*)outputBuf;
          } else {
            int localPeer = ncclShmem.comm.rankToLocalRank[peer];
            // coverity[deref_parm:FALSE] => work cannot be NULL if ipcReg != NULL
            exchgPtr = (T*)(work->coll.recvbuffOffset + work->coll.recvbuffRmtAddrs[localPeer]);
          }
          *slot = reinterpret_cast<void*>(exchgPtr);
        }
      }
      if (sendAcceptor) {
        int spins = 0;
        void* volatile* slot = ncclShmem.groups[group].sendConns[index]->ptrExchange;
        void* ptr;
        while (slot) {
          ptr = *slot;
          if (ptr != nullptr || checkAbort(flags, Aborted, spins)) break;
        }

        if (slot) {
          directBuff = reinterpret_cast<T*>(ptr);
          *slot = nullptr;
        } else {
          // coverity[var_deref_op]
          directBuff = (T*)work->dnOutputs[index];
        }
      }
      if (sendProvider) {
        int spins = 0;
        void* volatile* slot = ncclShmem.groups[group].sendConns[index]->ptrExchange;
        volatile uint64_t* argSlot0 = ncclShmem.groups[group].sendConns[index]->redOpArgExchange;
        volatile uint64_t* argSlot1 = ncclShmem.groups[group].sendConns[index]->redOpArgExchange + 1;
        // Wait for consumer to consume previous value before trampling it.
        if (slot && argSlot0 && argSlot1) {
          T* exchgPtr;
          while ((*slot != nullptr || *argSlot0 != 0 || *argSlot1 != 0) && !checkAbort(flags, Aborted, spins));
          // If there is no recv, then we are directly pulling from input buffer (e.g. directScatter)
          // Otherwise, we are pulling from output buffer (e.g. recvCopyDirectSend)
          directBuff = MaxRecv == 0 ? (T*)inputBuf : (T*)outputBuf;
          if (P2p) {
            exchgPtr = MaxRecv == 0 ? (T*)inputBuf : (T*)outputBuf;
          } else {
            int localPeer = ncclShmem.comm.rankToLocalRank[peer];
            if (MaxRecv == 0)
              // coverity[var_deref_op]
              exchgPtr = (T*)(work->coll.sendbuffOffset + work->coll.sendbuffRmtAddrs[localPeer]);
            else
              // coverity[var_deref_op]
              exchgPtr = (T*)(work->coll.recvbuffOffset + work->coll.recvbuffRmtAddrs[localPeer]);
          }

          // Exchange pre-scalers for use in direct pull
          *argSlot0 = (uint64_t(1) << 32) | (uint32_t)redOpArg;
          *argSlot1 = (uint64_t(1) << 32) | (uint32_t)(redOpArg >> 32);
          *slot = reinterpret_cast<T*>(exchgPtr);
        }
      }
      if (recvAcceptor) {
        int spins = 0;
        void* volatile* slot = ncclShmem.groups[group].recvConns[index]->ptrExchange;
        volatile uint64_t* argSlot0 = ncclShmem.groups[group].recvConns[index]->redOpArgExchange;
        volatile uint64_t* argSlot1 = ncclShmem.groups[group].recvConns[index]->redOpArgExchange + 1;
        void* ptr;
        while (slot) {
          ptr = *slot;
          if (ptr != nullptr || checkAbort(flags, Aborted, spins)) break;
        }

        if (slot && argSlot0 && argSlot1) {
          directBuff = reinterpret_cast<T*>(ptr);
          if (MaxSend != 0) { // reduce group rather than gather group
            // Store scalers for remote inputs
            uint64_t arg0, arg1;
            while (true) {
              arg0 = *argSlot0;
              arg1 = *argSlot1;
              if ((arg0 != 0 && arg1 != 0) || checkAbort(flags, Aborted, spins)) break;
            }
            ncclShmem.redOpArgs[1 + index] = ((arg1 & 0xffffffff) << 32) | (arg0 & 0xffffffff);
          }
          *argSlot0 = 0; *argSlot1 = 0;
          *slot = nullptr;
        } else {
          // Coverity complains about work being possibly NULL below.  However, slot
          // being NULL means that the NVLS buffer is registered (regUsed == 1)
          // so work can't be NULL in this code path.
          // coverity[var_deref_op]
          directBuff = (T*)work->dnInputs[index];
        }
      }
    }
  }

  __device__ void moveDataPtrs(intptr_t delta) {
    if (tid==0) {
      ncclShmem.groups[group].userInput = (T*)ncclShmem.groups[group].userInput + delta;
      ncclShmem.groups[group].userOutput = (T*)ncclShmem.groups[group].userOutput + delta;
    }
  }

  __device__ __forceinline__ void send(intptr_t inpIx, int eltN) {
    genericOp<0, 0, 0, 1, Input, -1>(inpIx, -1, eltN, false);
  }
  __device__ __forceinline__ void sendFromOutput(intptr_t outIx, int eltN) {
    genericOp<0, 0, 0, 1, Output, -1>(outIx, -1, eltN, false);
  }
  __device__ __forceinline__ void directSend(intptr_t inpIx, intptr_t outIx, int eltN) {
    genericOp<0, 1, 0, 1, Input, -1>(inpIx, outIx, eltN, false);
  }
  __device__ __forceinline__ void directSendFromOutput(intptr_t outIx, int eltN) {
    genericOp<0, 1, 0, 1, Output, -1>(outIx, outIx, eltN, false);
  }

  __device__ __forceinline__ void recv(intptr_t outIx, int eltN, bool postOp=false) {
    genericOp<0, 0, 1, 0, -1, Output>(-1, outIx, eltN, postOp);
  }
  __device__ __forceinline__ void directRecv(intptr_t outIx, int eltN, bool postOp=false) {
    genericOp<1, 0, 1, 0, -1, Output>(outIx, outIx, eltN, postOp);
  }
  __device__ __forceinline__ void directRecvCopy(intptr_t inpIx, intptr_t outIx, int eltN) {
    genericOp<1, 0, 1, 0, -1, Output>(inpIx, outIx, eltN, /*postOp=*/false);
  }

  __device__ __forceinline__ void copySend(intptr_t inpIx, intptr_t outIx, int eltN, bool postOp=false) {
    genericOp<0, 0, 0, 1, Input, Output>(inpIx, outIx, eltN, postOp);
  }
  __device__ __forceinline__ void directCopySend(intptr_t inpIx, intptr_t outIx, int eltN, bool postOp=false) {
    genericOp<0, 1, 0, 1, Input, Output>(inpIx, outIx, eltN, postOp);
  }

  __device__ __forceinline__ void recvSend(int eltN, bool postOp=false) {
    genericOp<0, 0, 1, 1, -1, -1>(-1, -1, eltN, postOp);
  }
  __device__ __forceinline__ void recvCopySend(intptr_t outIx, int eltN, bool postOp=false) {
    genericOp<0, 0, 1, 1, -1, Output>(-1, outIx, eltN, postOp);
  }
  __device__ __forceinline__ void directRecvCopyDirectSend(intptr_t inpIx, intptr_t outIx, int eltN, bool postOp=false) {
    genericOp<1, 1, 1, 1, -1, Output>(inpIx, outIx, eltN, postOp);
  }
  __device__ __forceinline__ void directRecvDirectSend(intptr_t inpIx, intptr_t outIx, int eltN, bool postOp=false) {
    genericOp<1, 1, 1, 1, -1, -1>(inpIx, outIx, eltN, postOp);
  }
  __device__ __forceinline__ void recvDirectSend(intptr_t outIx, int eltN, bool postOp=false) {
    genericOp<0, 1, 1, 1, -1, -1>(-1, outIx, eltN, postOp);
  }
  __device__ __forceinline__ void directRecvSend(intptr_t outIx, int eltN, bool postOp=false) {
    genericOp<1, 0, 1, 1, -1, -1>(outIx, outIx, eltN, postOp);
  }
  __device__ __forceinline__ void recvCopyDirectSend(intptr_t outIx, int eltN, bool postOp=false) {
    genericOp<0, 1, 1, 1, -1, Output>(-1, outIx, eltN, postOp);
  }

  __device__ __forceinline__ void recvReduceCopy(intptr_t inpIx, intptr_t outIx, int eltN, bool postOp=false) {
    genericOp<0, 0, 1, 0, Input, Output>(inpIx, outIx, eltN, postOp);
  }
  __device__ __forceinline__ void directRecvReduceCopy(intptr_t inpIx, intptr_t outIx, int eltN, bool postOp=false) {
    genericOp<1, 0, 1, 0, Input, Output>(inpIx, outIx, eltN, postOp);
  }

  __device__ __forceinline__ void recvReduceSend(intptr_t inpIx, int eltN, bool postOp=false) {
    genericOp<0, 0, 1, 1, Input, -1>(inpIx, -1, eltN, postOp);
  }
  __device__ __forceinline__ void directRecvReduceSend(intptr_t inpIx, int eltN, bool postOp=false) {
    genericOp<1, 0, 1, 1, Input, -1>(inpIx, -1, eltN, postOp);
  }
  __device__ __forceinline__ void recvReduceDirectSend(intptr_t inpIx, intptr_t outIx, int eltN, bool postOp=false) {
    genericOp<0, 1, 1, 1, Input, -1>(inpIx, outIx, eltN, postOp);
  }
  __device__ __forceinline__ void directRecvReduceDirectSend(intptr_t inpIx, intptr_t outIx, ssize_t eltN, bool postOp=false) {
    genericOp<1, 1, 1, 1, Input, -1>(inpIx, outIx, eltN, postOp);
  }

  __device__ __forceinline__ void recvReduceCopySend(intptr_t inpIx, intptr_t outIx, int eltN, bool postOp=false) {
    genericOp<0, 0, 1, 1, Input, Output>(inpIx, outIx, eltN, postOp);
  }
  __device__ __forceinline__ void recvReduceCopyDirectSend(intptr_t inpIx, intptr_t outIx, int eltN, bool postOp=false) {
    // Direct is only for the send part
    genericOp<0, 1, 1, 1, Input, Output>(inpIx, outIx, eltN, postOp);
  }
  __device__ __forceinline__ void directRecvReduceCopyDirectSend(intptr_t inpIx, intptr_t outIx, ssize_t eltN, bool postOp=false) {
    genericOp<1, 1, 1, 1, Input, Output>(inpIx, outIx, eltN, postOp);
  }

  __device__ __forceinline__ void
  scatter(intptr_t inpIx, ssize_t totalElem, int peerElem, ssize_t peerOffset, int skip, int shift) {
    ScatterGatherOp<0, 0, 0, 1>(inpIx, -1, totalElem, peerElem, peerOffset, skip, shift, /*postOp=*/false);
  }
  __device__ __forceinline__ void
  directScatter(intptr_t inpIx, ssize_t totalElem, int peerElem, ssize_t peerOffset, int skip, int shift) {
    ScatterGatherOp<0, 1, 0, 1>(inpIx, -1, totalElem, peerElem, peerOffset, skip, shift, /*postOp=*/false);
  }

  __device__ __forceinline__ void
  gather(intptr_t outIx, ssize_t totalElem, int peerElem, ssize_t peerOffset, int skip, int shift, bool postOp=false) {
    ScatterGatherOp<0, 0, 1, 0>(-1, outIx, totalElem, peerElem, peerOffset, skip, shift, postOp);
  }
  __device__ __forceinline__ void
  directGather(intptr_t outIx, ssize_t totalElem, int peerElem, ssize_t peerOffset, int skip, int shift) {
    ScatterGatherOp<1, 0, 1, 0>(-1, outIx, totalElem, peerElem, peerOffset, skip, shift, /*postOp=*/false);
  }

  __device__ __forceinline__ void patReduce(struct ncclPatStep* ps, struct ncclPatShmem* shmem) {
    if (ps->flags & PatSkipped) { patBarrier(); patBarrier(); return; } // Skipped
    int nelem = ps->nelem < 0 ? 0 : ps->nelem;
    T* userInput = (T*)ncclShmem.groups[group].userInput;
    T* userOutput = (T*)ncclShmem.groups[group].userOutput;

    bool recv = ps->recvDim >= 0 && (flags & (RolePostRecv|RoleWaitRecv));
    bool send = ps->sendDim >= 0 && (flags & (RolePostSend|RoleWaitSend));
    bool postRecv = ps->postRecv && recv;
    bool postSend = ps->postSend && send;
    struct ncclPatPeer* peer = NULL;
    if (recv) {
      peer = shmem->recvDims+ps->recvDim;
      step = peer->step;
    }
    if (send) {
      peer = shmem->sendDims+ps->sendDim;
      step = peer->step;
    }

    if (recv && (flags & RoleWaitRecv)) {
      ncclShmem.groups[group].srcs[0] = ((T*)peer->buff) + (step%NCCL_STEPS)*peer->connStepSize + ps->recvOffset;
      int spins = 0;
      while (peer->stepCache < step + StepPerSlice) {
        peer->stepCache = loadStepValue(peer->tailPtr);
        if (checkAbort(flags, Aborted, spins)) break;
      }
    }
    if (send && (flags & RoleWaitSend)) {
      int spins = 0;
      while (peer->stepCache + NCCL_STEPS < step + ps->stepOffset + StepPerSlice) {
        peer->stepCache = loadStepValue(peer->headPtr);
        if (checkAbort(flags, Aborted, spins)) break;
      }
      ncclShmem.groups[group].dsts[0] = ((T*)peer->buff) + ((step+ps->stepOffset)%NCCL_STEPS)*peer->connStepSize + ps->sendOffset;
      if (peer->accSize < ps->sendOffset + nelem + (step+ps->stepOffset)*peer->connStepSize) {
        // New data, add our own data to it.
        ncclShmem.groups[group].srcs[1] = userInput + ps->inpIx;
      } else {
        // There is already data in there, accumulate instead of writing to it.
        ncclShmem.groups[group].srcs[1] = ncclShmem.groups[group].dsts[0];
      }
    }
    long long int localAccSize = shmem->localAccSize;
    if (ps->sendDim < 0 && (flags & RoleOutput)) { // Destination is our own local buffer
      ncclShmem.groups[group].dsts[0] = userOutput + ps->outIx;
      if (localAccSize < ps->outIx + nelem) {
        // New data, add our own data to it.
        ncclShmem.groups[group].srcs[1] = userInput + ps->inpIx;
        localAccSize = ps->outIx + nelem;
      } else {
        // There is already data in there, accumulate instead of writing to it.
        ncclShmem.groups[group].srcs[1] = ncclShmem.groups[group].dsts[0];
      }
    }
    patBarrier();
    int nSrcs = 2;
    void** srcs = ncclShmem.groups[group].srcs;
    if (ps->recvDim < 0) { srcs++; nSrcs--; } // No peer to receive from, remove one source

    int workSize = ncclShmem.aborted ? 0 : nelem;

    reduceCopy<Unroll, RedOp, T, 0, 1, 2, 0, 1, 1, /*PreOpSrcs*/0>
      (tid, nthreads, ncclShmem.redOpArgs[0],  nullptr, /*postOp=*/false,
       nSrcs, srcs, 1, ncclShmem.groups[group].dsts, workSize);

    // Store conn step here inside the two barriers to make sure next reload will see the update.
    if (postSend && (flags & RolePostSend)) {
      if (peer->connFifo) {
        peer->connFifo[step%NCCL_STEPS].size = (ps->sendOffset + nelem)*sizeof(T);
      }
      peer->step = step += StepPerSlice;
      st_relaxed_sys_global(&peer->conn->step, step);
    }
    if (postRecv && (flags & RolePostRecv)) {
      peer->step = step += StepPerSlice;
      st_relaxed_sys_global(&peer->conn->step, step); // Also save in global mem for next op
    }

    // Update accSize
    if (ps->sendDim < 0 && (flags & RoleOutput)) atomicMax(&shmem->localAccSize, localAccSize);
    if (ps->sendDim >= 0 && (flags & RoleWaitSend)) atomicMax(&peer->accSize, ps->sendOffset + nelem + (step+ps->stepOffset)*peer->connStepSize);

    patBarrier();

    if (postSend && (flags & RolePostSend)) {
      if (nelem > 0 || peer->connFifo) fence_acq_rel_sys();
      st_relaxed_sys_global(peer->tailPtr, step);
    }
    if (postRecv && (flags & RolePostRecv)) {
      st_relaxed_sys_global(peer->headPtr, step);
    }
  }

  __device__ __forceinline__ void patCopy(struct ncclPatStep* ps, struct ncclPatShmem* shmem) {
    if (ps->flags & PatSkipped) { patBarrier(); patBarrier(); return; } // Skipped
    int nelem = ps->nelem < 0 ? 0 : ps->nelem;
    T* userInput = (T*)ncclShmem.groups[group].userInput;
    T* userOutput = (T*)ncclShmem.groups[group].userOutput;

    bool recv = ps->recvDim >= 0 && (flags & (RolePostRecv|RoleWaitRecv));
    bool send = ps->sendDim >= 0 && (flags & (RolePostSend|RoleWaitSend));
    bool postRecv = ps->postRecv && recv;
    bool postSend = ps->postSend && send;
    struct ncclPatPeer* peer = NULL;
    if (recv) {
      peer = shmem->recvDims+ps->recvDim;
      step = peer->step;
    }
    if (send) {
      peer = shmem->sendDims+ps->sendDim;
      step = peer->step;
    }

    if (recv && (flags & RoleWaitRecv)) {
      ncclShmem.groups[group].srcs[0] = ((T*)peer->buff) + ((step+ps->stepOffset)%NCCL_STEPS)*peer->connStepSize + ps->recvOffset;
      int spins = 0;
      while (peer->stepCache < step + ps->stepOffset + StepPerSlice) {
        peer->stepCache = loadStepValue(peer->tailPtr);
        if (checkAbort(flags, Aborted, spins)) break;
      }
      if (peer->accSize < ps->recvOffset + nelem + (step+ps->stepOffset)*peer->connStepSize) {
        // New data, copy to our output buffer.
        ncclShmem.groups[group].dsts[1] = userOutput + ps->outIx;
      } else {
        ncclShmem.groups[group].dsts[1] = ncclShmem.groups[group].srcs[0]; // Already done
      }
    }
    if (send && (flags & RoleWaitSend)) {
      int spins = 0;
      while (peer->stepCache + NCCL_STEPS < step + StepPerSlice) {
        peer->stepCache = loadStepValue(peer->headPtr);
        if (checkAbort(flags, Aborted, spins)) break;
      }
      ncclShmem.groups[group].dsts[0] = ((T*)peer->buff) + (step%NCCL_STEPS)*peer->connStepSize + ps->sendOffset;
    }
    long long int localAccSize = shmem->localAccSize;
    if (ps->recvDim < 0 && (flags & RoleInput)) { // Source is our own local buffer
      ncclShmem.groups[group].srcs[0] = userInput + ps->inpIx;
      if (localAccSize < ps->inpIx + nelem) {
        // New data, copy to our output buffer.
        ncclShmem.groups[group].dsts[1] = userOutput + ps->outIx;
        localAccSize = ps->inpIx + nelem;
      } else {
        // Already done
        ncclShmem.groups[group].dsts[1] = ncclShmem.groups[group].srcs[0];
      }
    }
    patBarrier();
    int nDsts = 2;
    void** dsts = ncclShmem.groups[group].dsts;
    if (ps->sendDim < 0) { dsts++; nDsts--; } // No peer to send to, remove one dest
    if (ncclShmem.groups[group].srcs[0] == ncclShmem.groups[group].dsts[1]) nDsts--; // In-place or already done.

    int workSize = ncclShmem.aborted ? 0 : nelem;

    reduceCopy<Unroll, RedOp, T, 0, 1, 1, 0, 1, 2, /*PreOpSrcs*/0>
      (tid, nthreads, ncclShmem.redOpArgs[0],  nullptr, /*postOp=*/false,
       1, ncclShmem.groups[group].srcs, nDsts, dsts, workSize);

    // Store conn step here inside the two barriers to make sure next reload will see the update.
    if (postSend && (flags & RolePostSend)) {
      if (peer->connFifo) {
        peer->connFifo[step%NCCL_STEPS].size = (ps->sendOffset + nelem)*sizeof(T);
      }
      peer->step = step += StepPerSlice;
      st_relaxed_sys_global(&peer->conn->step, step);
    }
    if (postRecv && (flags & RolePostRecv)) {
      peer->step = step += StepPerSlice;
      st_relaxed_sys_global(&peer->conn->step, step); // Also save in global mem for next op
    }

    // Update accSize
    if (ps->recvDim < 0 && (flags & RoleInput)) atomicMax(&shmem->localAccSize, localAccSize);
    if (ps->recvDim >= 0 && (flags & RoleWaitRecv)) atomicMax(&peer->accSize, ps->recvOffset + nelem + (step+ps->stepOffset)*peer->connStepSize);

    patBarrier();

    if (postSend && (flags & RolePostSend)) {
      if (nelem > 0 || peer->connFifo) fence_acq_rel_sys();
      st_relaxed_sys_global(peer->tailPtr, step);
    }
    if (postRecv && (flags & RolePostRecv)) {
      st_relaxed_sys_global(peer->headPtr, step);
    }
  }

};
