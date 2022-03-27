*******************************************************************
** Instructions pour l'utilisation des fichiers d'initialisation **
*******************************************************************


Conseils pour importer les fichiers dans Python:
------------------------------------------------

Utiliser la commande numpy.loadtxt.
ex: my_matrix = numpy.loadtxt('my_file.txt')
Vous pouvez également préciser que les données lues sont d'un certain type.
ex: my_matrix = numpy.loadtxt('my_file.txt', dtype = int)

La pas de discrétisation est précisé ci-dessous.


Conventions:
------------

Pour chaque cas à faire passer, 2 ou 3 matrices sont fournies:
	1) Dans tous les cas: une matrice représentant le domaine géométrique (appelée dom)
	Elle permet d'identifier
		1) Les noeuds qui ne doivent pas être calculés (valeur de 0). Une frange de 0 est placée
		autour de chaque domaine.
		2) Les noeuds internes prennent une valeur = 1.
		3) Les noeuds condition limite de Dirichlet sont repérés par une valeur de 2.
		
    	2) Dans tous les cas: une matrice qui donne un numéro de noeud pour chaque noeud de calcul (appelée num). Cette numérotation permet d'ordonner le système à résoudre. Pour plus d'infos, voir la séance introductive ou avec les étudiants-moniteurs.

	3) Pour le canal rectiligne (cas test): une matrice reprenant les valeurs des conditions de Dirichlet (appelée cl)
	
Informations complémentaires:
-----------------------------

1) Canal rectiligne - pas spatial = 0.5 m

2) Canal avec élargissement et rétrécissement - pas spatial = 0.01 m

3) Cas 2) avec îlot - pas spatial = 0.01 m (!attention! 2 CL de Dirichlet à tester)
