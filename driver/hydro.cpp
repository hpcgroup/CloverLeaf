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

#include "hydro.h"
#include "PdV.h"
#include "accelerate.h"
#include "advection.h"
#include "field_summary.h"
#include "flux_calc.h"
#include "reset_field.h"
#include "shared.h"
#include "timer.h"
#include "timestep.h"
#include "visit.h"

#include <vector>
#include <string>
#include <utility>

extern std::ostream g_out;
extern std::ostream csv_out;

int maxloc(const std::vector<double> &totals, const int len) {
  int loc = -1;
  double max = -1.0;
  for (int i = 0; i < len; ++i) {
    if (totals[i] >= max) {
      loc = i;
      max = totals[i];
    }
  }
  return loc;
}

void writeCSV(std::ostream &file,
    const std::vector<std::pair<std::string, double>> &times) {
  // Header
  for (const auto &p : times) {
    file << p.first << ",";
  }
  file << std::endl;

  // Timeing results, in milliseconds
  for (const auto &p : times) {
    file << 1000 * p.second << ",";
  }
  file << std::endl;
}

void hydro(global_variables &globals, parallel_ &parallel) {
  if (!globals.config.dumpDir.empty())
    clover::dump(globals, std::to_string(parallel.task) + "_" + std::to_string(globals.step) + "_05_hydro.txt");

  double timerstart = timer();
  bool profiler_off = globals.profiler_on;
  if (0 != globals.config.warmup_steps) {
    globals.profiler_on = false;
  }

  while (true) {
    if (globals.step == globals.config.warmup_steps) {
      globals.profiler_on = profiler_off;
      timerstart = timer();
      
      if (parallel.boss) {
        std::cout << "  " << globals.config.warmup_steps << " warmup steps completed" << std::endl;
      }
    }

    double step_time = timer();
    globals.step += 1;

    timestep(globals, parallel);
    if (!globals.config.dumpDir.empty())
      clover::dump(globals, std::to_string(parallel.task) + "_" + std::to_string(globals.step) + "_1_timestep.txt");

    PdV(globals, true);
    if (!globals.config.dumpDir.empty())
      clover::dump(globals, std::to_string(parallel.task) + "_" + std::to_string(globals.step) + "_2_PdV.txt");

    accelerate(globals);
    if (!globals.config.dumpDir.empty())
      clover::dump(globals, std::to_string(parallel.task) + "_" + std::to_string(globals.step) + "_3_accelerate.txt");

    PdV(globals, false);
    if (!globals.config.dumpDir.empty())
      clover::dump(globals, std::to_string(parallel.task) + "_" + std::to_string(globals.step) + "_4_PdV.txt");

    flux_calc(globals);
    if (!globals.config.dumpDir.empty())
      clover::dump(globals, std::to_string(parallel.task) + "_" + std::to_string(globals.step) + "_5_flux_calc.txt");

    advection(globals);
    if (!globals.config.dumpDir.empty())
      clover::dump(globals, std::to_string(parallel.task) + "_" + std::to_string(globals.step) + "_6_advection.txt");

    reset_field(globals);
    if (!globals.config.dumpDir.empty())
      clover::dump(globals, std::to_string(parallel.task) + "_" + std::to_string(globals.step) + "_7_reset_field.txt");

    globals.advect_x = !globals.advect_x;

    globals.time += globals.dt;
    //		globals.queue.wait_and_throw();

    if (globals.config.summary_frequency != 0) {
      if (globals.step % globals.config.summary_frequency == 0) field_summary(globals, parallel);
    }
    if (globals.config.visit_frequency != 0) {
      if (globals.step % globals.config.visit_frequency == 0) visit(globals, parallel);
    }

    // Sometimes there can be a significant start up cost that appears in the first step.
    // Sometimes it is due to the number of MPI tasks, or OpenCL kernel compilation.
    // On the short test runs, this can skew the results, so should be taken into account
    //  in recorded run times.
    double wall_clock{};
    double first_step = 0, second_step = 0;
    if (globals.step == 1) first_step = timer() - step_time;
    if (globals.step == 2) second_step = timer() - step_time;

    // JMK: simulation done
    if (globals.time + g_small > globals.config.end_time || globals.step >= globals.config.end_step) {

      globals.complete = true;
      field_summary(globals, parallel);
      if (globals.config.visit_frequency != 0) visit(globals, parallel);

      wall_clock = timer() - timerstart;
      if (parallel.boss) {
        g_out << std::endl
              << "Calculation complete" << std::endl
              << "Clover is finishing" << std::endl
              << "Wall clock " << wall_clock << std::endl
              << "First step overhead " << first_step - second_step << std::endl;
        std::cout << "  Wall clock " << wall_clock << std::endl //
                  << "  First step overhead " << first_step - second_step << std::endl;
      }

      std::vector<double> totals(parallel.max_task);
      if (globals.profiler_on) {
        // First we need to find the maximum kernel time for each task. This
        // seems to work better than finding the maximum time for each kernel and
        // adding it up, which always gives over 100%. I think this is because it
        // does not take into account compute overlaps before syncronisations
        // caused by halo exhanges.
        profiler_type &p = globals.profiler;
        double kernel_total = p.timestep + p.ideal_gas + p.viscosity + p.PdV + p.revert + p.acceleration + p.flux + p.cell_advection +
                              p.mom_advection + p.reset + p.summary + p.visit + p.tile_halo_exchange + p.self_halo_exchange +
                              p.mpi_halo_exchange;
        clover_allgather(kernel_total, totals);

        // So then what I do is use the individual kernel times for the
        // maximum kernel time task for the profile print
        int loc = maxloc(totals, parallel.max_task);
        kernel_total = totals[loc];

        clover_allgather(p.timestep, totals);
        p.timestep = totals[loc];
        clover_allgather(p.ideal_gas, totals);
        p.ideal_gas = totals[loc];
        clover_allgather(p.viscosity, totals);
        p.viscosity = totals[loc];
        clover_allgather(p.PdV, totals);
        p.PdV = totals[loc];
        clover_allgather(p.revert, totals);
        p.revert = totals[loc];
        clover_allgather(p.acceleration, totals);
        p.acceleration = totals[loc];
        clover_allgather(p.flux, totals);
        p.flux = totals[loc];
        clover_allgather(p.cell_advection, totals);
        p.cell_advection = totals[loc];
        clover_allgather(p.mom_advection, totals);
        p.mom_advection = totals[loc];
        clover_allgather(p.reset, totals);
        p.reset = totals[loc];
        clover_allgather(p.tile_halo_exchange, totals);
        p.tile_halo_exchange = totals[loc];
        clover_allgather(p.self_halo_exchange, totals);
        p.self_halo_exchange = totals[loc];
        clover_allgather(p.mpi_halo_exchange, totals);
        p.mpi_halo_exchange = totals[loc];
        clover_allgather(p.summary, totals);
        p.summary = totals[loc];
        clover_allgather(p.visit, totals);
        p.visit = totals[loc];
        clover_allgather(p.host_to_device, totals);
        p.host_to_device = totals[loc];
        clover_allgather(p.device_to_host, totals);
        p.device_to_host = totals[loc];
        
        if (parallel.boss) {
          double remainder = wall_clock - kernel_total - p.host_to_device - p.device_to_host;
          auto writeProfile = [&](auto &stream) {
            stream << std::fixed << std::endl
                   << " " << globals.config.warmup_steps << " warmup steps excluded from timing" << std::endl
                   << " Profiler Output        Time     Percentage" << std::endl
                   << " Timestep              :" << p.timestep << " " << 100.0 * (p.timestep / wall_clock) << std::endl
                   << " Ideal Gas             :" << p.ideal_gas << " " << 100.0 * (p.ideal_gas / wall_clock) << std::endl
                   << " Viscosity             :" << p.viscosity << " " << 100.0 * (p.viscosity / wall_clock) << std::endl
                   << " PdV                   :" << p.PdV << " " << 100.0 * (p.PdV / wall_clock) << std::endl
                   << " Revert                :" << p.revert << " " << 100.0 * (p.revert / wall_clock) << std::endl
                   << " Acceleration          :" << p.acceleration << " " << 100.0 * (p.acceleration / wall_clock) << std::endl
                   << " Fluxes                :" << p.flux << " " << 100.0 * (p.flux / wall_clock) << std::endl
                   << " Cell Advection        :" << p.cell_advection << " " << 100.0 * (p.cell_advection / wall_clock) << std::endl
                   << " Momentum Advection    :" << p.mom_advection << " " << 100.0 * (p.mom_advection / wall_clock) << std::endl
                   << " Reset                 :" << p.reset << " " << 100.0 * (p.reset / wall_clock) << std::endl
                   << " Summary               :" << p.summary << " " << 100.0 * (p.summary / wall_clock) << std::endl
                   << " Visit                 :" << p.visit << " " << 100.0 * (p.visit / wall_clock) << std::endl
                   << " Tile Halo Exchange    :" << p.tile_halo_exchange << " " << 100.0 * (p.tile_halo_exchange / wall_clock) << std::endl
                   << " Self Halo Exchange    :" << p.self_halo_exchange << " " << 100.0 * (p.self_halo_exchange / wall_clock) << std::endl
                   << " MPI Halo Exchange     :" << p.mpi_halo_exchange << " " << 100.0 * (p.mpi_halo_exchange / wall_clock) << std::endl
                   << " Total Kernel          :" << kernel_total << " " << 100.0 * (kernel_total / wall_clock) << std::endl
                   << " Host to Device        :" << p.host_to_device << " " << 100.0 * (p.host_to_device / wall_clock) << std::endl
                   << " Device to Host        :" << p.device_to_host << " " << 100.0 * (p.device_to_host / wall_clock) << std::endl
                   << " The Rest              :" << remainder << " " << 100.0 * remainder / wall_clock
                   << std::endl
                   << std::endl;
          };
          writeProfile(g_out);
          writeProfile(std::cout);

          if (globals.config.using_csv) {
            std::cout << "Saving timings to CSV\n" << std::endl;
            std::vector<std::pair<std::string, double>> cols;
            cols.emplace_back("timestep", p.timestep);
            cols.emplace_back("ideal_gas", p.ideal_gas);
            cols.emplace_back("viscosity", p.viscosity);
            cols.emplace_back("PdV", p.PdV);
            cols.emplace_back("revert", p.revert);
            cols.emplace_back("acceleration", p.acceleration);
            cols.emplace_back("fluxes", p.flux);
            cols.emplace_back("cell_advection", p.cell_advection);
            cols.emplace_back("mom_advection", p.mom_advection);
            cols.emplace_back("reset", p.reset);
            cols.emplace_back("summary", p.summary);
            cols.emplace_back("visits", p.visit);
            cols.emplace_back("tile_halo_exchange", p.tile_halo_exchange);
            cols.emplace_back("self_halo_exchange", p.self_halo_exchange);
            cols.emplace_back("mpi_halo_exchange", p.mpi_halo_exchange);
            cols.emplace_back("total_kernel", kernel_total);
            cols.emplace_back("host_to_device", p.host_to_device);
            cols.emplace_back("device_to_host", p.device_to_host);
            cols.emplace_back("other", remainder);
            cols.emplace_back("total", wall_clock);
            writeCSV(csv_out, cols);
          }
        }
      }

      // clover_finalize(); Skipped as just closes the file and calls MPI_Finalize (which is done back in main).

      break;
    }

    if (parallel.boss) {
      wall_clock = timer() - timerstart;
      double step_clock = timer() - step_time;
      g_out << "Wall clock " << wall_clock << std::endl;
      std::cout << "  Wall clock " << wall_clock << std::endl;
      double cells = globals.config.grid.x_cells * globals.config.grid.y_cells;
      double rstep = globals.step;
      if (globals.step > globals.config.warmup_steps) {
        rstep -= globals.config.warmup_steps;
      }
      double grind_time = wall_clock / (rstep * cells);
      double step_grind = step_clock / cells;
      std::cout << "  Average time per cell " << grind_time << std::endl;
      g_out << "Average time per cell " << grind_time << std::endl;
      std::cout << "  Step time per cell    " << step_grind << std::endl;
      g_out << "Step time per cell    " << step_grind << std::endl;
    }
  }
}
