import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import scipy.integrate as sci
from copy import deepcopy
import random
from typing import List, Tuple
import threading
import time

class RocketSimulation:
    def __init__(self):
        self.G = 6.6742e-11
        self.R_planet = 600000
        self.m_planet = 5.2915158e22
        self.max_thrust = 167970.0
        self.Isp1 = 250.0
        self.Isp2 = 400.0
        self.t_meco = 38.0
        self.t_sep = 2.0
        self.t2_start = 261.0
        self.t2_burn = 17.5
        self.mass0 = 5.3 * 2000 / 2.2
        self.mass1 = 0.2 * 2000 / 2.2
        self.period = 6000
        self.thrust_angle = 10.0
        
    def gravity(self, x, z):
        r = np.sqrt(x**2 + z**2)
        if r <= 0:
            return np.zeros(2)
        
        acc = self.G * self.m_planet / (r**3)
        return np.array([acc * x, acc * z])
    
    def propulsion(self, t):
        thrust = np.zeros(2)
        mdot = 0.0
        
        if t < self.t_meco:
            theta = np.radians(self.thrust_angle)
            ve = self.Isp1 * 9.81
            thrust = self.max_thrust * np.array([np.cos(theta), np.sin(theta)])
            mdot = -self.max_thrust / ve
            
        elif t < (self.t_meco + self.t_sep):
            mdot = -self.mass1 / self.t_sep
            
        elif self.t2_start < t < (self.t2_start + self.t2_burn):
            theta = np.radians(90)
            ve = self.Isp2 * 9.81
            thrust = self.max_thrust * np.array([np.cos(theta), np.sin(theta)])
            mdot = -self.max_thrust / ve
            
        return thrust, mdot
    
    def derivatives(self, state, t):
        x, z, vx, vz, mass = state
        
        if mass <= 0:
            return np.zeros(5)
        
        gravity_force = -self.gravity(x, z) * mass
        thrust_force, mdot = self.propulsion(t)
        
        acceleration = (gravity_force + thrust_force) / mass
        
        return np.array([vx, vz, acceleration[0], acceleration[1], mdot])
    
    def run_simulation(self):
        initial_state = np.array([
            self.R_planet,
            0.0,
            0.0,
            0.0,
            self.mass0
        ])
        
        t = np.linspace(0, self.period, 1000)
        
        states = sci.odeint(self.derivatives, initial_state, t)
        
        return t, states


class RocketGene:
    def __init__(self):
        self.thrust_angle = random.uniform(0, 90)
        self.t_meco = random.uniform(30, 50)
        self.t2_start = random.uniform(200, 300)
        self.t2_burn = random.uniform(10, 25)
        self.initial_mass = random.uniform(4000, 6000)
        self.max_thrust = random.uniform(150000, 180000)
        self.isp1 = random.uniform(240, 260)
        self.isp2 = random.uniform(380, 420)


class GeneticRocketOptimizer:
    def __init__(self, population_size: int = 50, callback=None):
        self.population_size = population_size
        self.population: List[RocketGene] = []
        self.target_altitude = 200000
        self.target_velocity = 7800
        self.callback = callback
        self.stop_flag = False
        
    def initialize_population(self):
        self.population = [RocketGene() for _ in range(self.population_size)]
    
    def calculate_fitness(self, gene: RocketGene) -> float:
        sim = RocketSimulation()
        
        sim.max_thrust = gene.max_thrust
        sim.Isp1 = gene.isp1
        sim.Isp2 = gene.isp2
        sim.t_meco = gene.t_meco
        sim.t2_start = gene.t2_start
        sim.t2_burn = gene.t2_burn
        sim.mass0 = gene.initial_mass
        sim.thrust_angle = gene.thrust_angle
        
        t, states = sim.run_simulation()
        
        final_x, final_z = states[-1, 0], states[-1, 1]
        final_vx, final_vz = states[-1, 2], states[-1, 3]
        final_mass = states[-1, 4]
        
        final_altitude = np.sqrt(final_x**2 + final_z**2) - sim.R_planet
        final_velocity = np.sqrt(final_vx**2 + final_vz**2)
        
        positions = np.column_stack((states[:, 0], states[:, 1]))
        radii = np.linalg.norm(positions, axis=1)
        orbit_circularity = 1 / (np.std(radii) + 1)
        
        altitude_error = abs(final_altitude - self.target_altitude)
        velocity_error = abs(final_velocity - self.target_velocity)
        
        mass_bonus = final_mass / gene.initial_mass
        
        fitness = (1000000 / 
                  (1 + altitude_error/10000 + velocity_error/100) 
                  * orbit_circularity 
                  * mass_bonus)
        
        return max(0, fitness)
    
    def select_parents(self) -> List[RocketGene]:
        fitnesses = [self.calculate_fitness(gene) for gene in self.population]
        
        def tournament_select():
            contestants = random.sample(list(enumerate(fitnesses)), 3)
            winner_idx = max(contestants, key=lambda x: x[1])[0]
            return self.population[winner_idx]
        
        return [tournament_select() for _ in range(self.population_size)]
    
    def crossover(self, parents: List[RocketGene]) -> List[RocketGene]:
        new_population = []
        
        for i in range(0, len(parents), 2):
            if i + 1 >= len(parents):
                new_population.append(deepcopy(parents[i]))
                continue
                
            parent1, parent2 = parents[i], parents[i + 1]
            child1, child2 = deepcopy(parent1), deepcopy(parent2)
            
            if random.random() < 0.7:
                for attr in ['thrust_angle', 't_meco', 't2_start', 't2_burn', 
                           'initial_mass', 'max_thrust', 'isp1', 'isp2']:
                    if random.random() < 0.5:
                        setattr(child1, attr, getattr(parent2, attr))
                        setattr(child2, attr, getattr(parent1, attr))
            
            new_population.extend([child1, child2])
        
        return new_population[:self.population_size]
    
    def mutate(self, population: List[RocketGene]) -> List[RocketGene]:
        for gene in population:
            if random.random() < 0.1:
                gene.thrust_angle += random.gauss(0, 5)
            if random.random() < 0.1:
                gene.t_meco += random.gauss(0, 2)
            if random.random() < 0.1:
                gene.t2_start += random.gauss(0, 10)
            if random.random() < 0.1:
                gene.t2_burn += random.gauss(0, 1)
            if random.random() < 0.1:
                gene.initial_mass *= (1 + random.gauss(0, 0.05))
            if random.random() < 0.1:
                gene.max_thrust *= (1 + random.gauss(0, 0.05))
            if random.random() < 0.1:
                gene.isp1 += random.gauss(0, 2)
            if random.random() < 0.1:
                gene.isp2 += random.gauss(0, 2)
        
        return population
    
    def evolve(self, generations: int = 50, delay: float = 0.5) -> Tuple[RocketGene, float]:
        self.initialize_population()
        
        best_fitness = 0
        best_gene = None
        
        fitness_history = []
        best_genes_history = []
        
        self.stop_flag = False
        self.generations = generations
        
        for gen in range(generations):
            if self.stop_flag:
                break
                
            parents = self.select_parents()
            
            new_population = self.crossover(parents)
            
            new_population = self.mutate(new_population)
            
            self.population = new_population
            
            fitness_scores = [self.calculate_fitness(gene) for gene in self.population]
            current_best_idx = np.argmax(fitness_scores)
            current_best = self.population[current_best_idx]
            current_best_fitness = fitness_scores[current_best_idx]
            
            fitness_history.append(current_best_fitness)
            best_genes_history.append(deepcopy(current_best))
            
            if current_best_fitness > best_fitness:
                best_fitness = current_best_fitness
                best_gene = deepcopy(current_best)
            
            if self.callback:
                self.callback(gen, current_best, current_best_fitness, best_gene, best_fitness, 
                             fitness_history, best_genes_history)
            
            time.sleep(delay)
        
        return best_gene, best_fitness


class RocketOptimizationGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Rocket Optimization")
        self.root.geometry("1200x800")
        
        self.create_frames()
        
        self.create_input_widgets()
        
        self.create_plot_area()
        
        self.create_trajectory_area()
        
        self.create_results_display()
        
        self.optimizer = GeneticRocketOptimizer(callback=self.update_display)
        
        self.opt_thread = None
        
        self.fitness_history = []
        self.best_genes_history = []
        self.current_gen = 0
        self.animation_speed = 0.3
        self.animating = False
        
        self.root.bind("<Configure>", self.on_window_resize)
    
    def create_frames(self):
        self.left_frame = ttk.Frame(self.root)
        self.left_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        self.right_frame = ttk.Frame(self.root)
        self.right_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=2)
        self.root.grid_rowconfigure(0, weight=1)
        
        self.input_frame = ttk.LabelFrame(self.left_frame, text="Optimization Parameters")
        self.input_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        self.results_frame = ttk.LabelFrame(self.left_frame, text="Results")
        self.results_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        self.plot_frame = ttk.LabelFrame(self.right_frame, text="Fitness Progress")
        self.plot_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        self.trajectory_frame = ttk.LabelFrame(self.right_frame, text="Trajectory Evolution")
        self.trajectory_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        self.animation_frame = ttk.Frame(self.right_frame)
        self.animation_frame.pack(fill="x", expand=False, padx=5, pady=5)
    
    def create_input_widgets(self):
        ttk.Label(self.input_frame, text="Population Size:").grid(row=0, column=0, sticky="w", padx=5, pady=2)
        self.pop_size_var = tk.StringVar(value="50")
        ttk.Entry(self.input_frame, textvariable=self.pop_size_var, width=10).grid(row=0, column=1, padx=5, pady=2)
        
        ttk.Label(self.input_frame, text="Generations:").grid(row=1, column=0, sticky="w", padx=5, pady=2)
        self.gen_var = tk.StringVar(value="50")
        ttk.Entry(self.input_frame, textvariable=self.gen_var, width=10).grid(row=1, column=1, padx=5, pady=2)
        
        ttk.Label(self.input_frame, text="Target Altitude (km):").grid(row=2, column=0, sticky="w", padx=5, pady=2)
        self.target_alt_var = tk.StringVar(value="200")
        ttk.Entry(self.input_frame, textvariable=self.target_alt_var, width=10).grid(row=2, column=1, padx=5, pady=2)
        
        ttk.Label(self.input_frame, text="Target Velocity (m/s):").grid(row=3, column=0, sticky="w", padx=5, pady=2)
        self.target_vel_var = tk.StringVar(value="7800")
        ttk.Entry(self.input_frame, textvariable=self.target_vel_var, width=10).grid(row=3, column=1, padx=5, pady=2)
        
        ttk.Label(self.input_frame, text="Delay Between Generations (s):").grid(row=4, column=0, sticky="w", padx=5, pady=2)
        self.delay_var = tk.StringVar(value="0.3")
        ttk.Entry(self.input_frame, textvariable=self.delay_var, width=10).grid(row=4, column=1, padx=5, pady=2)
        
        ttk.Separator(self.input_frame, orient="horizontal").grid(row=5, column=0, columnspan=2, sticky="ew", pady=10)
        ttk.Label(self.input_frame, text="Planet Parameters", font="-weight bold").grid(row=6, column=0, columnspan=2, sticky="w", padx=5, pady=2)
        
        ttk.Label(self.input_frame, text="Planet Radius (km):").grid(row=7, column=0, sticky="w", padx=5, pady=2)
        self.planet_radius_var = tk.StringVar(value="600")
        ttk.Entry(self.input_frame, textvariable=self.planet_radius_var, width=10).grid(row=7, column=1, padx=5, pady=2)
        
        ttk.Label(self.input_frame, text="Planet Mass (kg):").grid(row=8, column=0, sticky="w", padx=5, pady=2)
        self.planet_mass_var = tk.StringVar(value="5.29e22")
        ttk.Entry(self.input_frame, textvariable=self.planet_mass_var, width=10).grid(row=8, column=1, padx=5, pady=2)
        
        ttk.Separator(self.input_frame, orient="horizontal").grid(row=9, column=0, columnspan=2, sticky="ew", pady=10)
        ttk.Label(self.input_frame, text="Rocket Parameter Ranges", font="-weight bold").grid(row=10, column=0, columnspan=2, sticky="w", padx=5, pady=2)
        
        ttk.Label(self.input_frame, text="Mass Range (kg):").grid(row=11, column=0, sticky="w", padx=5, pady=2)
        self.mass_min_var = tk.StringVar(value="4000")
        self.mass_max_var = tk.StringVar(value="6000")
        mass_frame = ttk.Frame(self.input_frame)
        mass_frame.grid(row=11, column=1, padx=5, pady=2)
        ttk.Entry(mass_frame, textvariable=self.mass_min_var, width=5).pack(side="left")
        ttk.Label(mass_frame, text="-").pack(side="left")
        ttk.Entry(mass_frame, textvariable=self.mass_max_var, width=5).pack(side="left")
        
        ttk.Label(self.input_frame, text="Thrust Range (N):").grid(row=12, column=0, sticky="w", padx=5, pady=2)
        self.thrust_min_var = tk.StringVar(value="150000")
        self.thrust_max_var = tk.StringVar(value="180000")
        thrust_frame = ttk.Frame(self.input_frame)
        thrust_frame.grid(row=12, column=1, padx=5, pady=2)
        ttk.Entry(thrust_frame, textvariable=self.thrust_min_var, width=8).pack(side="left")
        ttk.Label(thrust_frame, text="-").pack(side="left")
        ttk.Entry(thrust_frame, textvariable=self.thrust_max_var, width=8).pack(side="left")
        
        btn_frame = ttk.Frame(self.input_frame)
        btn_frame.grid(row=13, column=0, columnspan=2, sticky="ew", pady=10)
        
        ttk.Button(btn_frame, text="Start Optimization", command=self.start_optimization).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="Stop", command=self.stop_optimization).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="Reset", command=self.reset_gui).pack(side="left", padx=5)
    
    def create_plot_area(self):
        self.fig_fitness = plt.Figure(figsize=(6, 3), dpi=100, tight_layout=True)
        self.ax_fitness = self.fig_fitness.add_subplot(111)
        self.canvas_fitness = FigureCanvasTkAgg(self.fig_fitness, master=self.plot_frame)
        canvas_widget = self.canvas_fitness.get_tk_widget()
        canvas_widget.pack(fill="both", expand=True)
        
        self.ax_fitness.set_title("Best Fitness per Generation")
        self.ax_fitness.set_xlabel("Generation")
        self.ax_fitness.set_ylabel("Fitness")
        self.ax_fitness.grid(True)
        self.fitness_line, = self.ax_fitness.plot([], [], 'b-')
        self.canvas_fitness.draw()
    
    def create_trajectory_area(self):
        self.fig_traj = plt.Figure(figsize=(6, 6), dpi=100, tight_layout=True)
        self.ax_traj = self.fig_traj.add_subplot(111)
        self.canvas_traj = FigureCanvasTkAgg(self.fig_traj, master=self.trajectory_frame)
        canvas_widget = self.canvas_traj.get_tk_widget()
        canvas_widget.pack(fill="both", expand=True)
        
        self.ax_traj.set_title("Rocket Trajectory")
        self.ax_traj.set_aspect('equal')
        self.ax_traj.grid(True)
        
        ttk.Label(self.animation_frame, text="Generation:").pack(side="left", padx=5)
        self.gen_slider = ttk.Scale(self.animation_frame, from_=0, to=1, orient="horizontal", command=self.update_trajectory_display)
        self.gen_slider.pack(side="left", fill="x", expand=True, padx=5)
        
        self.gen_label = ttk.Label(self.animation_frame, text="0/0")
        self.gen_label.pack(side="left", padx=5)
        
        self.animate_btn = ttk.Button(self.animation_frame, text="▶ Play", command=self.toggle_animation)
        self.animate_btn.pack(side="left", padx=5)
    
    def create_results_display(self):
        self.results_text = tk.Text(self.results_frame, height=20, width=40)
        self.results_text.pack(fill="both", expand=True, padx=5, pady=5)
    
    def on_window_resize(self, event):
        if event.widget == self.root:
            self.root.after(100, self.redraw_figures)
    
    def redraw_figures(self):
        self.fig_fitness.tight_layout()
        self.canvas_fitness.draw()
        
        self.fig_traj.tight_layout()
        self.ax_traj.set_aspect('equal')
        self.canvas_traj.draw()
    
    def update_display(self, gen, current_best, current_fitness, overall_best, best_fitness, 
                      fitness_history, best_genes_history):
        self.root.after(0, lambda: self._update_display_safe(gen, current_best, current_fitness, 
                                                          overall_best, best_fitness,
                                                          fitness_history, best_genes_history))
    
    def _update_display_safe(self, gen, current_best, current_fitness, overall_best, best_fitness,
                           fitness_history, best_genes_history):
        self.fitness_line.set_data(range(len(fitness_history)), fitness_history)
        self.ax_fitness.relim()
        self.ax_fitness.autoscale_view()
        self.fig_fitness.tight_layout()
        self.canvas_fitness.draw()
        
        self.gen_label.config(text=f"{gen+1}/{self.optimizer.generations}")
        self.gen_slider.config(to=max(1, len(best_genes_history)))
        self.gen_slider.set(gen+1)
        
        self.fitness_history = fitness_history.copy()
        self.best_genes_history = best_genes_history.copy()
        
        self.update_trajectory_display(None)
        
        self.update_results_text(gen, current_best, current_fitness, overall_best, best_fitness)
    
    def update_trajectory_display(self, event=None):
        if not self.best_genes_history:
            return
        
        try:
            gen_idx = int(self.gen_slider.get()) - 1
        except:
            gen_idx = 0
            
        gen_idx = max(0, min(gen_idx, len(self.best_genes_history) - 1))
        self.current_gen = gen_idx
        
        self.gen_label.config(text=f"{gen_idx+1}/{len(self.best_genes_history)}")
        
        gene = self.best_genes_history[gen_idx]
        
        sim = RocketSimulation()
        
        try:
            sim.R_planet = float(self.planet_radius_var.get()) * 1000
            sim.m_planet = float(self.planet_mass_var.get())
        except ValueError:
            pass
        
        sim.max_thrust = gene.max_thrust
        sim.Isp1 = gene.isp1
        sim.Isp2 = gene.isp2
        sim.t_meco = gene.t_meco
        sim.t2_start = gene.t2_start
        sim.t2_burn = gene.t2_burn
        sim.mass0 = gene.initial_mass
        sim.thrust_angle = gene.thrust_angle
        
        t, states = sim.run_simulation()
        
        x, z = states[:, 0], states[:, 1]
        
        self.ax_traj.clear()
        
        self.ax_traj.plot(x, z, 'r-', label='Orbit')
        self.ax_traj.plot(x[0], z[0], 'g*', label='Launch')
        
        theta = np.linspace(0, 2*np.pi, 100)
        x_planet = sim.R_planet * np.sin(theta)
        y_planet = sim.R_planet * np.cos(theta)
        self.ax_traj.plot(x_planet, y_planet, 'b-', label='Planet')
        
        self.ax_traj.set_aspect('equal')
        self.ax_traj.grid(True)
        self.ax_traj.legend()
        self.ax_traj.set_title(f"Trajectory - Generation {gen_idx+1}")
        
        self.fig_traj.tight_layout()
        
        self.canvas_traj.draw()
    
    def update_results_text(self, gen, current_best, current_fitness, overall_best, best_fitness):
        self.results_text.delete(1.0, tk.END)
        
        self.results_text.insert(tk.END, f"==== GENERATION {gen+1} ====\n\n")
        
        self.results_text.insert(tk.END, "CURRENT BEST:\n")
        self.results_text.insert(tk.END, f"Fitness: {current_fitness:.2f}\n")
        self.results_text.insert(tk.END, f"Thrust Angle: {current_best.thrust_angle:.2f}°\n")
        self.results_text.insert(tk.END, f"MECO: {current_best.t_meco:.2f}s\n")
        self.results_text.insert(tk.END, f"2nd Stage Start: {current_best.t2_start:.2f}s\n")
        self.results_text.insert(tk.END, f"2nd Stage Burn: {current_best.t2_burn:.2f}s\n")
        self.results_text.insert(tk.END, f"Initial Mass: {current_best.initial_mass:.2f}kg\n")
        self.results_text.insert(tk.END, f"Max Thrust: {current_best.max_thrust:.2f}N\n")
        self.results_text.insert(tk.END, f"Isp1: {current_best.isp1:.2f}s\n")
        self.results_text.insert(tk.END, f"Isp2: {current_best.isp2:.2f}s\n\n")
        
        self.results_text.insert(tk.END, "OVERALL BEST:\n")
        if overall_best:
            self.results_text.insert(tk.END, f"Fitness: {best_fitness:.2f}\n")
            self.results_text.insert(tk.END, f"Thrust Angle: {overall_best.thrust_angle:.2f}°\n")
            self.results_text.insert(tk.END, f"MECO: {overall_best.t_meco:.2f}s\n")
            self.results_text.insert(tk.END, f"2nd Stage Start: {overall_best.t2_start:.2f}s\n")
            self.results_text.insert(tk.END, f"2nd Stage Burn: {overall_best.t2_burn:.2f}s\n")
            self.results_text.insert(tk.END, f"Initial Mass: {overall_best.initial_mass:.2f}kg\n")
            self.results_text.insert(tk.END, f"Max Thrust: {overall_best.max_thrust:.2f}N\n")
            self.results_text.insert(tk.END, f"Isp1: {overall_best.isp1:.2f}s\n")
            self.results_text.insert(tk.END, f"Isp2: {overall_best.isp2:.2f}s\n")

    def start_optimization(self):
        if self.opt_thread and self.opt_thread.is_alive():
            messagebox.showinfo("Already Running", "Optimization is already running.")
            return
        
        try:
            pop_size = int(self.pop_size_var.get())
            generations = int(self.gen_var.get())
            target_alt = float(self.target_alt_var.get()) * 1000
            target_vel = float(self.target_vel_var.get())
            delay = float(self.delay_var.get())
            
            planet_radius = float(self.planet_radius_var.get()) * 1000
            planet_mass = float(self.planet_mass_var.get())
            
            mass_min = float(self.mass_min_var.get())
            mass_max = float(self.mass_max_var.get())
            thrust_min = float(self.thrust_min_var.get())
            thrust_max = float(self.thrust_max_var.get())
            
        except ValueError:
            messagebox.showerror("Input Error", "Please enter valid numerical values for all parameters.")
            return
            
        self.optimizer = GeneticRocketOptimizer(population_size=pop_size, callback=self.update_display)
        self.optimizer.target_altitude = target_alt
        self.optimizer.target_velocity = target_vel
        self.optimizer.generations = generations
        
        original_init = RocketGene.__init__
        
        def new_init(self_gene):
            self_gene.thrust_angle = random.uniform(0, 90)
            self_gene.t_meco = random.uniform(30, 50)
            self_gene.t2_start = random.uniform(200, 300)
            self_gene.t2_burn = random.uniform(10, 25)
            self_gene.initial_mass = random.uniform(mass_min, mass_max)
            self_gene.max_thrust = random.uniform(thrust_min, thrust_max)
            self_gene.isp1 = random.uniform(240, 260)
            self_gene.isp2 = random.uniform(380, 420)
        original_init = RocketGene.__init__
        
        def new_init(self_gene):
            self_gene.thrust_angle = random.uniform(0, 90)
            self_gene.t_meco = random.uniform(30, 50)
            self_gene.t2_start = random.uniform(200, 300)
            self_gene.t2_burn = random.uniform(10, 25)
            self_gene.initial_mass = random.uniform(mass_min, mass_max)
            self_gene.max_thrust = random.uniform(thrust_min, thrust_max)
            self_gene.isp1 = random.uniform(240, 260)
            self_gene.isp2 = random.uniform(380, 420)
        
        RocketGene.__init__ = new_init
        
        # Override RocketSimulation parameters
        original_init_sim = RocketSimulation.__init__
        
        def new_init_sim(self_sim):
            original_init_sim(self_sim)
            self_sim.R_planet = planet_radius
            self_sim.m_planet = planet_mass
            
        RocketSimulation.__init__ = new_init_sim
        
        # Reset animation controls
        self.fitness_history = []
        self.best_genes_history = []
        self.gen_slider.set(0)
        self.gen_label.config(text="0/0")
        
        # Start optimization in thread
        self.opt_thread = threading.Thread(target=self.optimizer.evolve, 
                                         args=(generations, delay))
        self.opt_thread.daemon = True
        self.opt_thread.start()
    
    def stop_optimization(self):
        """Stop the ongoing optimization"""
        if self.optimizer:
            self.optimizer.stop_flag = True
            messagebox.showinfo("Stopping", "Optimization will stop after current generation.")
            
        if self.animating:
            self.toggle_animation()
    
    def reset_gui(self):
        """Reset all GUI elements to initial state"""
        # Stop optimization if running
        self.stop_optimization()
        
        # Clear plots
        self.ax_fitness.clear()
        self.ax_fitness.set_title("Best Fitness per Generation")
        self.ax_fitness.set_xlabel("Generation")
        self.ax_fitness.set_ylabel("Fitness")
        self.ax_fitness.grid(True)
        self.fitness_line, = self.ax_fitness.plot([], [], 'b-')
        self.canvas_fitness.draw()
        
        self.ax_traj.clear()
        self.ax_traj.set_title("Rocket Trajectory")
        self.ax_traj.set_aspect('equal')
        self.ax_traj.grid(True)
        self.canvas_traj.draw()
        
        # Reset slider and label
        self.gen_slider.set(0)
        self.gen_label.config(text="0/0")
        
        # Clear results text
        self.results_text.delete(1.0, tk.END)
        
        # Reset history
        self.fitness_history = []
        self.best_genes_history = []
        
        # Reset animation
        self.animating = False
        self.animate_btn.config(text="▶ Play")
    
    def toggle_animation(self):
        """Toggle animation of trajectory evolution"""
        if self.animating:
            # Stop animation
            self.animating = False
            self.animate_btn.config(text="▶ Play")
        else:
            # Start animation
            if not self.best_genes_history:
                messagebox.showinfo("No Data", "No optimization data available for animation.")
                return
                
            self.animating = True
            self.animate_btn.config(text="⏸ Pause")
            self.animate_trajectory()
    
    def animate_trajectory(self):
        """Animate through trajectory evolution"""
        if not self.animating:
            return
            
        # Increment generation
        total_gens = len(self.best_genes_history)
        if total_gens <= 0:
            self.toggle_animation()
            return
            
        next_gen = (self.current_gen + 1) % total_gens
        self.gen_slider.set(next_gen + 1)  # +1 because slider uses 1-based indexing
        
        # Schedule next frame
        self.root.after(int(self.animation_speed * 1000), self.animate_trajectory)


def run_simulation_demo():
    """Run a demonstration of the rocket simulation"""
    # Create and run a basic simulation
    sim = RocketSimulation()
    t, states = sim.run_simulation()
    
    # Plot the trajectory
    plt.figure(figsize=(10, 10))
    
    # Extract position coordinates
    x, z = states[:, 0], states[:, 1]
    
    # Plot trajectory
    plt.plot(x, z, 'r-', label='Trajectory')
    plt.plot(x[0], z[0], 'go', label='Launch')
    
    # Plot planet
    theta = np.linspace(0, 2*np.pi, 100)
    x_planet = sim.R_planet * np.sin(theta)
    y_planet = sim.R_planet * np.cos(theta)
    plt.plot(x_planet, y_planet, 'b-', label='Planet')
    
    # Mark key events on the trajectory
    meco_idx = np.argmin(np.abs(t - sim.t_meco))
    sep_idx = np.argmin(np.abs(t - (sim.t_meco + sim.t_sep)))
    t2_start_idx = np.argmin(np.abs(t - sim.t2_start))
    t2_end_idx = np.argmin(np.abs(t - (sim.t2_start + sim.t2_burn)))
    
    plt.plot(x[meco_idx], z[meco_idx], 'ro', label='MECO')
    plt.plot(x[sep_idx], z[sep_idx], 'bo', label='Stage Separation')
    plt.plot(x[t2_start_idx], z[t2_start_idx], 'mo', label='2nd Stage Ignition')
    plt.plot(x[t2_end_idx], z[t2_end_idx], 'ko', label='2nd Stage Cutoff')
    
    # Configure plot
    plt.title('Rocket Trajectory Simulation')
    plt.xlabel('X Position (m)')
    plt.ylabel('Z Position (m)')
    plt.grid(True)
    plt.axis('equal')
    plt.legend()
    
    plt.show()
    
    return sim, t, states


def run_optimization_demo():
    """Run a demonstration of the genetic optimization"""
    optimizer = GeneticRocketOptimizer(population_size=20)
    best_gene, best_fitness = optimizer.evolve(generations=10)
    
    print(f"Best fitness: {best_fitness}")
    print(f"Best gene parameters:")
    print(f"  Thrust angle: {best_gene.thrust_angle:.2f} degrees")
    print(f"  MECO time: {best_gene.t_meco:.2f} s")
    print(f"  2nd stage start: {best_gene.t2_start:.2f} s")
    print(f"  2nd stage burn: {best_gene.t2_burn:.2f} s")
    print(f"  Initial mass: {best_gene.initial_mass:.2f} kg")
    print(f"  Max thrust: {best_gene.max_thrust:.2f} N")
    print(f"  Isp1: {best_gene.isp1:.2f} s")
    print(f"  Isp2: {best_gene.isp2:.2f} s")
    
    # Run simulation with best parameters
    sim = RocketSimulation()
    sim.thrust_angle = best_gene.thrust_angle
    sim.t_meco = best_gene.t_meco
    sim.t2_start = best_gene.t2_start
    sim.t2_burn = best_gene.t2_burn
    sim.mass0 = best_gene.initial_mass
    sim.max_thrust = best_gene.max_thrust
    sim.Isp1 = best_gene.isp1
    sim.Isp2 = best_gene.isp2
    
    t, states = sim.run_simulation()
    
    # Plot trajectory
    plt.figure(figsize=(10, 10))
    
    # Extract position coordinates
    x, z = states[:, 0], states[:, 1]
    
    # Plot trajectory
    plt.plot(x, z, 'r-', label='Optimized Trajectory')
    plt.plot(x[0], z[0], 'go', label='Launch')
    
    # Plot planet
    theta = np.linspace(0, 2*np.pi, 100)
    x_planet = sim.R_planet * np.sin(theta)
    y_planet = sim.R_planet * np.cos(theta)
    plt.plot(x_planet, y_planet, 'b-', label='Planet')
    
    # Configure plot
    plt.title('Optimized Rocket Trajectory')
    plt.xlabel('X Position (m)')
    plt.ylabel('Z Position (m)')
    plt.grid(True)
    plt.axis('equal')
    plt.legend()
    
    plt.show()
    
    return best_gene, best_fitness, sim, t, states


if __name__ == "__main__":
    # Run GUI application
    root = tk.Tk()
    app = RocketOptimizationGUI(root)
    root.mainloop()
    
    # Uncomment to run demos
    # run_simulation_demo()
    # run_optimization_demo()