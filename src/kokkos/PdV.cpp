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

#include "PdV.h"
#include "comms.h"
#include "ideal_gas.h"
#include "report.h"
#include "revert.h"
#include "timer.h"
#include "sync.h"
#include "update_halo.h"

//  @brief Fortran PdV kernel.
//  @author Wayne Gaudin
//  @details Calculates the change in energy and density in a cell using the
//  change on cell volume due to the velocity gradients in a cell. The time
//  level of the velocity data depends on whether it is invoked as the
//  predictor or corrector.
void PdV_kernel(bool predict, int x_min, int x_max, int y_min, int y_max, double dt, Kokkos::View<double **> &xarea,
                Kokkos::View<double **> &yarea, Kokkos::View<double **> &volume, Kokkos::View<double **> &density0,
                Kokkos::View<double **> &density1, Kokkos::View<double **> &energy0, Kokkos::View<double **> &energy1,
                Kokkos::View<double **> &pressure, Kokkos::View<double **> &viscosity, Kokkos::View<double **> &xvel0,
                Kokkos::View<double **> &xvel1, Kokkos::View<double **> &yvel0, Kokkos::View<double **> &yvel1,
                Kokkos::View<double **> &volume_change) {

  // DO k=y_min,y_max
  //   DO j=x_min,x_max
  Kokkos::MDRangePolicy<Kokkos::Rank<2>> policy({x_min + 1, y_min + 1}, {x_max + 2, y_max + 2});

  if (predict) {

    Kokkos::parallel_for(
        "PdV predict=true", policy, KOKKOS_LAMBDA(const int j, const int k) {
          double left_flux = (xarea(j, k) * (xvel0(j, k) + xvel0(j, k + 1) + xvel0(j, k) + xvel0(j, k + 1))) * 0.25 * dt * 0.5;
          double right_flux =
              (xarea(j + 1, k) * (xvel0(j + 1, k) + xvel0(j + 1, k + 1) + xvel0(j + 1, k) + xvel0(j + 1, k + 1))) * 0.25 * dt * 0.5;
          double bottom_flux = (yarea(j, k) * (yvel0(j, k) + yvel0(j + 1, k) + yvel0(j, k) + yvel0(j + 1, k))) * 0.25 * dt * 0.5;
          double top_flux =
              (yarea(j, k + 1) * (yvel0(j, k + 1) + yvel0(j + 1, k + 1) + yvel0(j, k + 1) + yvel0(j + 1, k + 1))) * 0.25 * dt * 0.5;
          double total_flux = right_flux - left_flux + top_flux - bottom_flux;
          double volume_change_s = volume(j, k) / (volume(j, k) + total_flux);
          double recip_volume = 1.0 / volume(j, k);
          double energy_change = (pressure(j, k) / density0(j, k) + viscosity(j, k) / density0(j, k)) * total_flux * recip_volume;
          energy1(j, k) = energy0(j, k) - energy_change;
          density1(j, k) = density0(j, k) * volume_change_s;
        });

  } else {

    Kokkos::parallel_for(
        "PdV predict=false", policy, KOKKOS_LAMBDA(const int j, const int k) {
          double left_flux = (xarea(j, k) * (xvel0(j, k) + xvel0(j, k + 1) + xvel1(j, k) + xvel1(j, k + 1))) * 0.25 * dt;
          double right_flux =
              (xarea(j + 1, k) * (xvel0(j + 1, k) + xvel0(j + 1, k + 1) + xvel1(j + 1, k) + xvel1(j + 1, k + 1))) * 0.25 * dt;
          double bottom_flux = (yarea(j, k) * (yvel0(j, k) + yvel0(j + 1, k) + yvel1(j, k) + yvel1(j + 1, k))) * 0.25 * dt;
          double top_flux = (yarea(j, k + 1) * (yvel0(j, k + 1) + yvel0(j + 1, k + 1) + yvel1(j, k + 1) + yvel1(j + 1, k + 1))) * 0.25 * dt;
          double total_flux = right_flux - left_flux + top_flux - bottom_flux;
          double volume_change_s = volume(j, k) / (volume(j, k) + total_flux);
          double recip_volume = 1.0 / volume(j, k);
          double energy_change = (pressure(j, k) / density0(j, k) + viscosity(j, k) / density0(j, k)) * total_flux * recip_volume;
          energy1(j, k) = energy0(j, k) - energy_change;
          density1(j, k) = density0(j, k) * volume_change_s;
        });
  }
}

//  @brief Driver for the PdV update.
//  @author Wayne Gaudin
//  @details Invokes the user specified kernel for the PdV update.
void PdV(global_variables &globals, bool predict) {

  double kernel_time{};
  if (globals.profiler_on) kernel_time = timer();

  globals.error_condition = 0;

  for (int tile = 0; tile < globals.config.tiles_per_chunk; ++tile) {
    tile_type &t = globals.chunk.tiles[tile];
    PdV_kernel(predict, t.info.t_xmin, t.info.t_xmax, t.info.t_ymin, t.info.t_ymax, globals.dt, t.field.xarea.view, t.field.yarea.view,
               t.field.volume.view, t.field.density0.view, t.field.density1.view, t.field.energy0.view, t.field.energy1.view,
               t.field.pressure.view, t.field.viscosity.view, t.field.xvel0.view, t.field.xvel1.view, t.field.yvel0.view,
               t.field.yvel1.view, t.field.work_array1.view);
  }

  clover_check_error(globals.error_condition);
  if (globals.profiler_on) {
    if (globals.should_sync_profile) clover_sync();
    globals.profiler.PdV += timer() - kernel_time;
  }

  if (globals.error_condition == 1) {
    report_error((char *)"PdV", (char *)"error in PdV");
  }

  if (predict) {
    if (globals.profiler_on) kernel_time = timer();
    for (int tile = 0; tile < globals.config.tiles_per_chunk; ++tile) {
      ideal_gas(globals, tile, true);
    }

    if (globals.profiler_on) {
      if (globals.should_sync_profile) clover_sync();
      globals.profiler.ideal_gas += timer() - kernel_time;
    }

    int fields[NUM_FIELDS];
    for (int i = 0; i < NUM_FIELDS; ++i)
      fields[i] = 0;
    fields[field_pressure] = 1;
    update_halo(globals, fields, 1);
  }

  if (predict) {
    if (globals.profiler_on) kernel_time = timer();
    revert(globals);
    if (globals.profiler_on) {
      if (globals.should_sync_profile) clover_sync();
      globals.profiler.revert += timer() - kernel_time;
    }
  }
}
