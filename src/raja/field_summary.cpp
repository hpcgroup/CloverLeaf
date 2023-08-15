/*
 Crown Copyright 2012 AWE.

 This file is part of CloverLeaf.

 CloverLeaf is free software: you can redistribute it and/or modify it under
 the terms of the GNU General Public License as published by the
 Free Software Foundation, either version 3 of the License, or (at your option)
 any later version.

 CloverLeaf is distributed in the hope that it will be useful, but
 WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
 details.

 You should have received a copy of the GNU General Public License along with
 CloverLeaf. If not, see http://www.gnu.org/licenses/.
 */

#include "field_summary.h"
#include "context.h"
#include "ideal_gas.h"
#include "report.h"
#include "timer.h"

#include <iomanip>
#include <numeric>

#include <RAJA/RAJA.hpp>

//  @brief Fortran field summary kernel
//  @author Wayne Gaudin
//  @details The total mass, internal energy, kinetic energy and volume weighted
//  pressure for the chunk is calculated.
//  @brief Driver for the field summary kernels
//  @author Wayne Gaudin
//  @details The user specified field summary kernel is invoked here. A summation
//  across all mesh chunks is then performed and the information outputed.
//  If the run is a test problem, the final result is compared with the expected
//  result and the difference output.
//  Note the reference solution is the value returned from an Intel compiler with
//  ieee options set on a single core crun.

void field_summary(global_variables &globals, parallel_ &parallel) {

  clover_report_step_header(globals, parallel);

  double kernel_time = 0;
  if (globals.profiler_on) kernel_time = timer();

  for (int tile = 0; tile < globals.config.tiles_per_chunk; ++tile) {
    ideal_gas(globals, tile, false);
  }

  if (globals.profiler_on) {
    clover::checkError(cudaDeviceSynchronize());
    globals.profiler.ideal_gas += timer() - kernel_time;
    kernel_time = timer();
  }

  double vol = 0.0;
  double mass = 0.0;
  double ie = 0.0;
  double ke = 0.0;
  double press = 0.0;

  const int BLOCK = 256;
  clover::Buffer1D<double> vol_buffer(globals.context, BLOCK);
  clover::Buffer1D<double> mass_buffer(globals.context, BLOCK);
  clover::Buffer1D<double> ie_buffer(globals.context, BLOCK);
  clover::Buffer1D<double> ke_buffer(globals.context, BLOCK);
  clover::Buffer1D<double> press_buffer(globals.context, BLOCK);

  for (int tile = 0; tile < globals.config.tiles_per_chunk; ++tile) {
    tile_type &t = globals.chunk.tiles[tile];

    int ymax = t.info.t_ymax;
    int ymin = t.info.t_ymin;
    int xmax = t.info.t_xmax;
    int xmin = t.info.t_xmin;
    field_type &field = t.field;
    using EXEC_POL3   = RAJA::cuda_exec<RAJA_BLOCK_SIZE>;
    using REDUCE_POL3 = RAJA::cuda_reduce;

    RAJA::ReduceSum<REDUCE_POL3, double> RAJAvol(0);
    RAJA::ReduceSum<REDUCE_POL3, double> RAJAmass(0);
    RAJA::ReduceSum<REDUCE_POL3, double> RAJAie(0);
    RAJA::ReduceSum<REDUCE_POL3, double> RAJAke(0);
    RAJA::ReduceSum<REDUCE_POL3, double> RAJApress(0);

    int range = (ymax - ymin + 1) * (xmax - xmin + 1);
//    clover::par_reduce<BLOCK, BLOCK>([=] __device__(int gid) {
    RAJA::forall<EXEC_POL3>(RAJA::TypedRangeSegment<int>(0, range),
        [=] RAJA_DEVICE (int gid) {
      int v = gid;

      const size_t j = xmin + 1 + v % (xmax - xmin + 1);
      const size_t k = ymin + 1 + v / (xmax - xmin + 1);
      double vsqrd = 0.0;
      for (size_t kv = k; kv <= k + 1; ++kv) {
        for (size_t jv = j; jv <= j + 1; ++jv) {
          vsqrd += 0.25 * (field.xvel0(jv, kv) * field.xvel0(jv, kv) + field.yvel0(jv, kv) * field.yvel0(jv, kv));
        }
      }
      double cell_vol = field.volume(j, k);
      double cell_mass = cell_vol * field.density0(j, k);

      RAJAvol += cell_vol;
      RAJAmass += cell_mass;
      RAJAie  += cell_mass * field.energy0(j, k);
      RAJAke += cell_mass * 0.5 * vsqrd;
      RAJApress += cell_vol * field.pressure(j, k);
    });
    vol = RAJAvol.get();
    mass = RAJAmass.get();
    ie = RAJAie.get();
    ke = RAJAke.get();
    press = RAJApress.get();
  }
  vol_buffer.release();
  mass_buffer.release();
  ie_buffer.release();
  ke_buffer.release();
  press_buffer.release();

  //    summary s;
  //
  //    for (int tile = 0; tile < globals.config.tiles_per_chunk; ++tile) {
  //      tile_type &t = globals.chunk.tiles[tile];
  //
  //      int ymax = t.info.t_ymax;
  //      int ymin = t.info.t_ymin;
  //      int xmax = t.info.t_xmax;
  //      int xmin = t.info.t_xmin;
  //      field_type &field = t.field;
  //
  //      auto r = clover::range<size_t>(0, (ymax - ymin + 1) * (xmax - xmin + 1));
  //      s = std::transform_reduce(r.begin(), r.end(), s, std::plus<>(), [=](size_t idx) {
  //        const size_t j = xmin + 1 + idx % (xmax - xmin + 1);
  //        const size_t k = ymin + 1 + idx / (xmax - xmin + 1);
  //        double vsqrd = 0.0;
  //        for (size_t kv = k; kv <= k + 1; ++kv) {
  //          for (size_t jv = j; jv <= j + 1; ++jv) {
  //            vsqrd += 0.25 * (field.xvel0(jv, kv) * field.xvel0(jv, kv) + field.yvel0(jv, kv) * field.yvel0(jv, kv));
  //          }
  //        }
  //        double cell_vol = field.volume(j, k);
  //        double cell_mass = cell_vol * field.density0(j, k);
  //
  //        return summary{.vol = cell_vol,
  //                       .mass = cell_mass,
  //                       .ie = cell_mass * field.energy0(j, k),
  //                       .ke = cell_mass * 0.5 * vsqrd,
  //                       .press = cell_vol * field.pressure(j, k)};
  //      });
  //    }
  //
  //    auto [vol, mass, ie, ke, press] = s;

  clover_sum(vol);
  clover_sum(mass);
  clover_sum(ie);
  clover_sum(ke);
  clover_sum(press);

  if (globals.profiler_on) {
    clover::checkError(cudaDeviceSynchronize());
    globals.profiler.summary += timer() - kernel_time;
  }

  clover_report_step(globals, parallel, vol, mass, ie, ke, mass);
}
