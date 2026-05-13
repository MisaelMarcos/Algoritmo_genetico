import datetime
import cv2
import math
import random
import copy
import numpy as np
import sys
import os
import concurrent.futures

# =====================================================================
# 1. LEITURA DO ARQUIVO TSP
# =====================================================================
def carregar_distancias_tsp(caminho_arquivo="brazil58.tsp", qtde_cidades=58):
    distancias = {}
    try:
        with open(caminho_arquivo, "r") as objArq:
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
# 2. CLASSES DO ALGORITMO GENÉTICO
# =====================================================================
class City:
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
    def __init__(self, h, w, no_cities, dist_matrix, points=None):
        self.h = h
        self.w = w
        self.no_cities = no_cities
        self.dist_matrix = dist_matrix 
        self.cities = []
        for city in range(self.no_cities):
            self.cities.append(City(self, city, points[city] if points is not None else None))

    def get_distance(self, pos1, pos2):
        if pos1 == pos2:
            return 0
        return self.dist_matrix[(pos1, pos2)]

class Worker:
    def __init__(self, mapp):
        cities = np.arange(mapp.no_cities)
        random.shuffle(cities)
        self.route = list(cities)
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

def breed_families(families, best_only=False):
    for i, x in enumerate(families):
        if i >= 1:
            if best_only:
                families[i].breed(mummy=families[int(random.random() * len(families))].best_worker)
            else:
                families[i].breed(mummy=families[i - 1].best_worker)
    return families

# =====================================================================
# 3. FUNÇÃO RUN (COM SISTEMA DE LOGS ARQUIVADOS)
# =====================================================================
def run(caminho_arquivo, run_id=1, families_count=5, workers_count=200, breed_rate=20, max_time_seconds=120):
    best_dist = 99999999
    run_count = 0
    
    qtde_cidades = 58 
    dist_matrix = carregar_distancias_tsp(caminho_arquivo, qtde_cidades)

    maps = Map(800, 800, qtde_cidades, dist_matrix, None)
    families = [Family(workers_count, maps) for x in range(families_count)]

    time_start = datetime.datetime.utcnow()
    
    # Prepara o arquivo de log para esta execução específica
    arquivo_log = f"logs_simulacoes/log_execucao_{run_id:02d}.txt"
    with open(arquivo_log, "w", encoding="utf-8") as f:
        f.write(f"=== INICIANDO SIMULAÇÃO {run_id:02d} ===\n")
        f.write(f"Início: {time_start.strftime('%H:%M:%S')}\n")
        f.write("Acompanhamento:\n\n")

    while True:
        _ = [fam.sort_workers() for fam in families]
        families = sorted(families, key=lambda item: item.best_score)
        best_score = families[0].best_score

        if best_score < best_dist:
            best_dist = best_score
            # Sempre que achar um recorde novo absoluto, anota no log imediatamente
            time_now = datetime.datetime.utcnow()
            elapsed_seconds = (time_now - time_start).total_seconds()
            with open(arquivo_log, "a", encoding="utf-8") as f:
                f.write(f"[{elapsed_seconds:05.1f}s] NOVO RECORDE | Geração: {run_count} | Distância: {best_dist}\n")

        run_count += 1

        if run_count % breed_rate == 0:
            families = breed_families(families)

        # Atualiza o log a cada 100 gerações para mostrar que o processo não travou (Batimento Cardíaco)
        if run_count % 100 == 0:
            time_now = datetime.datetime.utcnow()
            elapsed_seconds = (time_now - time_start).total_seconds()
            with open(arquivo_log, "a", encoding="utf-8") as f:
                f.write(f"[{elapsed_seconds:05.1f}s] Status       | Geração: {run_count} | Melhor Dist Atual: {best_dist}\n")

        # CONDIÇÃO DE PARADA
        time_now = datetime.datetime.utcnow()
        elapsed_seconds = (time_now - time_start).total_seconds()
        
        if elapsed_seconds >= max_time_seconds:
            # Finaliza o log
            with open(arquivo_log, "a", encoding="utf-8") as f:
                f.write(f"\n=== SIMULAÇÃO FINALIZADA ===\n")
                f.write(f"Total de Gerações: {run_count}\n")
                f.write(f"Melhor Distância Final: {best_dist}\n")
                f.write(f"Rota Encontrada: {families[0].best_worker.route}\n")

            return {
                "id_execucao": run_id,
                "melhor_distancia": best_dist,
                "geracoes_calculadas": run_count,
                "tempo_gasto_segundos": elapsed_seconds,
                "melhor_rota": families[0].best_worker.route
            }

# =====================================================================
# 4. EXECUTOR EM LOTE
# =====================================================================
def iniciar_bateria_de_testes(quantidade_testes=10, tempo_maximo_segundos=120):
    # Cria a pasta de logs se ela não existir
    os.makedirs("logs_simulacoes", exist_ok=True)
    
    print(f"Iniciando {quantidade_testes} execuções simultâneas do Algoritmo Genético...")
    print(f"Cada execução vai rodar por exatamente {tempo_maximo_segundos} segundos.\n")
    print(">>> DICA: Abra a pasta 'logs_simulacoes' e abra os arquivos .txt para acompanhar o progresso em tempo real!\n")
    
    resultados_finais = []
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=quantidade_testes) as executor:
        futuros = [
            executor.submit(run, "brazil58.tsp", i+1, 5, 200, 20, tempo_maximo_segundos) 
            for i in range(quantidade_testes)
        ]
        
        for futuro in concurrent.futures.as_completed(futuros):
            try:
                resultado = futuro.result()
                resultados_finais.append(resultado)
                print(f"[OK] Execução {resultado['id_execucao']:02d} finalizada! Distância: {resultado['melhor_distancia']} | Gravado em: log_execucao_{resultado['id_execucao']:02d}.txt")
            except Exception as e:
                print(f"[ERRO] Falha em uma das execuções: {e}")

    print("\n" + "="*50)
    print("RESUMO DE TODAS AS EXECUÇÕES")
    print("="*50)
    
    resultados_finais = sorted(resultados_finais, key=lambda x: x['melhor_distancia'])
    
    for res in resultados_finais:
        print(f"Execução {res['id_execucao']:02d} | Distância: {res['melhor_distancia']} | Gerações: {res['geracoes_calculadas']}")

    print(f"\n✅ A MELHOR ROTA GERAL FOI DA EXECUÇÃO {resultados_finais[0]['id_execucao']:02d}")
    print(f"Rota salva nos logs e mantida em memória para uso.")
    
    return resultados_finais

if __name__ == "__main__":
    lista_de_informacoes = iniciar_bateria_de_testes(quantidade_testes=10, tempo_maximo_segundos=120)