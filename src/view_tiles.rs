//! A module for splitting the view into tiles that can be rendered separately.
//! This is a way to split the work among multiple processors, and is great
//! because it groups pixels that are close and that are therefore likely to
//! make accesses to the same objects during ray tracing, and because it looks
//! cool.

pub struct Tile {
    pub x_min: u16,
    pub y_min: u16,
    pub x_max: u16,
    pub y_max: u16,
}

pub struct Tiles {
    view_width: u16,
    view_height: u16,
    tile_size: u16,
    columns: u16,
    tile_count: u16,
}

impl Tiles {
    pub fn new(view_width: u16, view_height: u16, tile_size: u16) -> Tiles {
        let columns = (view_width + tile_size - 1) / tile_size;
        let rows = (view_height + tile_size - 1) / tile_size;
        Tiles {
            view_width,
            view_height,
            tile_size,
            columns,
            tile_count: columns * rows,
        }
    }

    pub fn get(&self, i: u16) -> Tile {
        assert!(i < self.tile_count, "i is larger than the tile count");

        let row = i / self.columns;
        let column = i % self.columns;

        let x_min = column * self.tile_size;
        let y_min = row * self.tile_size;
        let x_max = (x_min + self.tile_size).min(self.view_width);
        let y_max = (y_min + self.tile_size).min(self.view_height);

        Tile {
            x_min,
            y_min,
            x_max,
            y_max,
        }
    }

    pub fn tile_count(&self) -> u16 {
        self.tile_count
    }
}
