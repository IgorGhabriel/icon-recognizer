Needed code files to run:

with python

save-database.py - to create the database with the icons paths and characteristcs;

main.py - to input an icon and show the ones similars from the database;

with jupyter notebbok

salvando-em-database.ipynb - to create the database with the icons paths and characteristcs;

comparador.ipynb - to input an icon and show the ones similars from the database;

To create the database, a folder named icons-base with the name of the module, on it, folders with the style of the icons, on them, the .svg icons;

Examples:

icons-base/arcticons/default/icons-name.svg

icons-base/some-icon-module/sharp/icons-name.svg
    
icons-base/some-icon-module/two-tone/icons-name.svg
    

if the module doesnt have a style, name it 'default';

running save-database.py/salvando-em-database.ipynb will create a folder 'icons-png', with the same path style from icons-base, and a .sqlite file, named 'icons-database.sqlite'

