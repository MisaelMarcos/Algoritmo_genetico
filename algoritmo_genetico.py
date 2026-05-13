import datetime
import cv2
import math
import random
import copy
import numpy as np
import sys

# =====================================================================
# 1. INTEGRAÇÃO: LEITURA DO ARQUIVO TSP (Do seu Código 1)
# =====================================================================
def carregar_distancias_tsp(caminho_arquivo="brazil58.tsp", qtde_cidades=58):
    distancias = {}
    try:
        with open(caminho_arquivo, "r") as objArq:
            # Adaptando para índices de 0 a 57 (para ser compatível com as listas do GA)
            for i in range(qtde_cidades - 1): 
                linha = objArq.readline()
                lista = linha.split()
                for j in range(i + 1, qtde_cidades):
                    if len(lista) > 0:
                        peso = int(lista.pop(0))
                        distancias[(i, j)] = peso
                        distancias[(j, i)] = peso
                    else:
                        print(f"Erro! linha {i+1} do arquivo não possui elementos suficientes")
                        sys.exit()
        return distancias
    except FileNotFoundError:
        print(f"Erro: Arquivo '{caminho_arquivo}' não encontrado no diretório atual.")
        sys.exit()

# =====================================================================
# 2. CLASSES DO ALGORITMO GENÉTICO (Do seu Código 2 - Adaptadas)
# =====================================================================
class City:
    """ Guarda as coordenadas fictícias das cidades apenas para o desenho na tela """
    def __init__(self, maps, iden, point):
        self.iden = iden
        if point:
            self.x = point[0]
            self.y = point[1]
        else:
            self.x = random.randint(0, maps.w)
            self.y = random.randint(0, maps.h)

    def get_position(self):
        return self.y, self.x

class Map:
    """ Guarda os detalhes do mapa e as DISTÂNCIAS REAIS do arquivo TSP """
    def __init__(self, h, w, no_cities, dist_matrix, points=None):
        self.h = h
        self.w = w
        self.no_cities = no_cities
        self.dist_matrix = dist_matrix # <--- Matriz de distâncias integrada aqui
        self.cities = []
        
        for city in range(self.no_cities):
            self.cities.append(City(self, city, points[city] if points is not None else None))

    def get_distance(self, pos1, pos2):
        # INTEGRAÇÃO: Agora busca a distância real do arquivo em vez de usar coordenadas X,Y
        if pos1 == pos2:
            return 0
        return self.dist_matrix[(pos1, pos2)]

class Worker:
    def __init__(self, mapp):
        cities = np.arange(mapp.no_cities)
        random.shuffle(cities)
        self.route = cities
        self.mapp = mapp

    def get_distance(self, add=True):
        dist = []
        for i, x in enumerate(self.route):
            if i == len(self.route) - 1:
                dist.append(self.mapp.get_distance(x, self.route[0]))
            else:
                dist.append(self.mapp.get_distance(x, self.route[i + 1]))

        if add:
            return sum(dist)
        else:
            return dist

    def get_genes(self, worker1):
        geneA = int(random.random() * len(self.route))
        geneB = int(random.random() * len(self.route))
        start = min(geneA, geneB)
        end = max(geneA, geneB)

        childP1 = []
        for i in range(start, end):
            childP1.append(self.route[i])

        childP2 = [x for x in worker1.route if x not in childP1]
        return childP1 + childP2

    def mutate(self, daddy_route):
        self.route = copy.copy(daddy_route)
        if random.randint(1, 100) == 1:
            random.shuffle(self.route)
        else:
            for i, x in enumerate(self.route):
                if random.randint(0, 100) <= 5:
                    current = self.route[i]
                    nextt = current
                    while nextt != current:
                        nextt = random.randint(0, len(self.route) - 1)

                    self.route[i] = self.route[nextt]
                    self.route[nextt] = current
        return self

class Family:
    def __init__(self, workers, maps):
        self.workers = [Worker(maps) for x in range(workers)]
        self.eliete = []
        self.best_score = 99999999

    def sort_workers(self):
        self.workers = sorted(self.workers, key=lambda item: item.get_distance())
        eliete = int(len(self.workers) * 0.02)
        if eliete < 3:
            eliete = 4
        self.eliete = self.workers[:eliete]
        self.best_score = self.eliete[0].get_distance()
        self.best_worker = self.eliete[0]
        self.breed()

    def breed(self, mummy=None):
        random.shuffle(self.eliete)
        if mummy is None:
            the_best, the_second_best = self.eliete[:2]
        else:
            the_best = self.best_worker
            the_second_best = mummy

        best_five_perc = self.workers[0:int(len(self.workers) * 0.05)]
        [work.mutate(the_best.get_genes(the_second_best)) for work in self.workers if work not in best_five_perc]

# =====================================================================
# 3. FUNÇÕES DE VISUALIZAÇÃO E UTILIDADE
# =====================================================================
def get_cities(maps):
    board = np.ones((maps.h, maps.w, 3), dtype=np.uint8) * 255
    for city in maps.cities:
        board = cv2.circle(board, (city.y, city.x), radius=int(maps.w * 0.01), color=(125, 125, 125), thickness=-1)
    return board

def show_best(board, best, maps):
    num_cols = len(best.route)
    blue = np.linspace(100, 250, num_cols)
    green = np.linspace(50, 200, num_cols)
    red = np.linspace(200, 50, num_cols)

    for i, x in enumerate(best.route):
        point1 = maps.cities[x].get_position()
        if i == len(maps.cities) - 1:
            point2 = maps.cities[best.route[0]].get_position()
        else:
            point2 = maps.cities[best.route[i + 1]].get_position()
        board = cv2.line(board, point1, point2, color=(int(blue[i]), int(green[i]), int(red[i])), thickness=3)

    cv2.imshow("Algoritmo Genetico - TSP Brasil 58", board)

def get_round_points(no):
    """ Gera coordenadas circulares fictícias apenas para podermos desenhar """
    point_center = (400, 400)
    radius = 350
    points_list = []
    for x in np.linspace(0, 2 * math.pi, no + 1):
        px = int(point_center[0] + radius * math.cos(x))
        py = int(point_center[1] + radius * math.sin(x))
        points_list.append([px, py])
    return points_list[:-1]

def strfdelta(tdelta, fmt):
    d = {"days": tdelta.days}
    d["hours"], rem = divmod(tdelta.seconds, 3600)
    d["minutes"], d["seconds"] = divmod(rem, 60)
    return fmt.format(**d)

def breed_families(families, best_only=False):
    for i, x in enumerate(families):
        if i >= 1:
            if best_only:
                families[i].breed(mummy=families[int(random.random() * len(families))].best_worker)
            else:
                families[i].breed(mummy=families[i - 1].best_worker)
    return families

# =====================================================================
# 4. LOOP PRINCIPAL DE EXECUÇÃO
# =====================================================================
def run(caminho_arquivo, families_count=5, workers_count=200, breed_rate=20, display=True):
    best_dist = 99999999
    run_count = 0
    
    # 1. Carrega as distâncias do seu arquivo TSP
    print("Carregando arquivo TSP...")
    qtde_cidades = 58 # Fixo conforme seu código original
    dist_matrix = carregar_distancias_tsp(caminho_arquivo, qtde_cidades)

    # 2. Cria pontos circulares fictícios para a visualização gráfica
    points = get_round_points(qtde_cidades)
    maps = Map(800, 800, qtde_cidades, dist_matrix, points)

    # 3. Carrega as famílias
    families = [Family(workers_count, maps) for x in range(families_count)]

    best_text = f"Melhor distância : {best_dist} na geracao # {run_count}"
    run_text = ""

    if display:
        print("Display carregado - clique na janela do mapa e pressione 'q' para sair.")
        board_base = get_cities(maps)
    else:
        print("Pressione 'ctrl + c' no terminal para sair.")
    print("----------------------------------------------")

    time_start = datetime.datetime.utcnow()

    while True:
        if display:
            board = board_base.copy()

        _ = [fam.sort_workers() for fam in families]
        families = sorted(families, key=lambda item: item.best_score)
        best_score = families[0].best_score

        if best_score < best_dist:
            best_dist = best_score
            best_text = f"Melhor distância: {families[0].best_worker.get_distance()} na geracao # {run_count}"

        run_count += 1

        if run_count % breed_rate == 0:
            families = breed_families(families)

        run_text = f" Geracao atual # {run_count - 1}"

        if display:
            show_best(board, families[0].best_worker, maps)
            if cv2.waitKey(1) == ord('q'):
                cv2.destroyAllWindows()
                break

        time_now = datetime.datetime.utcnow()
        time_taken_text = "  -  Tempo total: " + strfdelta((time_now - time_start), "{hours:02d}:{minutes:02d}:{seconds:02d}")

        print("\r                                                                                         ", end="")
        print("\r" + best_text + " | " + run_text + time_taken_text, end="")

if __name__ == "__main__":
    # Ajuste os parâmetros aqui conforme achar necessário
    ARQUIVO_TSP = "brazil58.tsp"
    FAMILIAS = 5
    TRABALHADORES = 200
    TAXA_CRUZAMENTO = 20
    DESENHAR = True

    def iniciar():
        run(ARQUIVO_TSP, FAMILIAS, TRABALHADORES, TAXA_CRUZAMENTO, DESENHAR)
    iniciar()